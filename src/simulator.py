"""Newton/Warp helpers for settling particles inside a cylindrical container."""

from dataclasses import dataclass
import json
import math
import trimesh
from pathlib import Path

import numpy as np
from tqdm import tqdm
import warp as wp
import newton


type Vec3 = tuple[float, float, float]
type Vec4 = tuple[float, float, float, float]


@dataclass
class SimulationResult:
    """Final rigid-body pose for one simulated object.

    Attributes:
        obj_id: External object identifier supplied when the body was added.
        body_id: Internal Newton body index.
        translation_xyz: Final body translation in Newton x, y, z order.
        rotation_xyzw: Final body orientation as a quaternion in x, y, z, w order.
    """

    obj_id: int
    body_id: int
    translation_xyz: Vec3
    rotation_xyzw: Vec4

    @property
    def translation_zyx(self) -> Vec3:
        """Return translation reordered for volume arrays that use z, y, x axes."""
        x, y, z = self.translation_xyz
        return (z, y, x)

    def to_dict(self) -> dict:
        """Convert the dataclass to a JSON-serializable dictionary."""
        return {
            "obj_id": int(self.obj_id),
            "body_id": int(self.body_id),
            "translation_xyz": [float(v) for v in self.translation_xyz],
            "rotation_xyzw": [float(v) for v in self.rotation_xyzw],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SimulationResult":
        return cls(
            obj_id=int(data["obj_id"]),
            body_id=int(data["body_id"]),
            translation_xyz=tuple(float(v) for v in data["translation_xyz"]),
            rotation_xyzw=tuple(float(v) for v in data["rotation_xyzw"]),
        )



class NewtonSimulator:
    """Build, run, and inspect a Newton rigid-body packing simulation.

    The simulator owns a single Newton model. Add static geometry and dynamic
    bodies first, then call :meth:`finalize`, then :meth:`simulate`.
    """

    def __init__(
        self,
        *,
        device: str | None = None,
        viewer: newton.viewer.ViewerBase | None = None
    ):
        """Create an empty simulation builder.

        Args:
            device: Warp/Newton device string such as ``"cuda:0"``. ``None``
                lets Warp choose its default device.
            viewer: Optional Newton viewer used for frame and contact logging.
        """
        wp.init()

        self.device = device
        self.builder = newton.ModelBuilder()
        self.model = None
        self.state_0 = None
        self.state_1 = None
        self.solver = None
        self.sim_time = 0.0
        self.viewer = viewer

        self.container_radius = None
        self.container_height = None

        self._finalized = False
        self._body_id_to_obj_id = {}


    def _ensure_not_finalized(self):
        """Reject builder mutations after the Newton model has been finalized."""
        if self._finalized:
            raise RuntimeError("Model has already been finalised.")


    def add_cylinder(
        self,
        radius: float,
        height: float,
        *,
        wall_thickness: float = 0.3,
        floor_thickness: float = 1.0,
        n_faces: int = 48,
        center: Vec3 = (0.0, 0.0, -1.0),
        static_cfg: newton.ModelBuilder.ShapeConfig | None = None,
    ) -> None:
        """Add an open-top static cylindrical container with a floor.

        Args:
            radius: Inner radius of the container.
            height: Wall height measured from the floor base.
            wall_thickness: Radial thickness of the box segments used for walls.
            floor_thickness: Thickness of the bottom cylinder.
            n_faces: Number of box segments approximating the circular wall.
            center: Floor origin in Newton x, y, z coordinates.
            static_cfg: Optional shared collision/material config for the floor
                and wall segments.

        Newton uses x, y, z coordinates here. The floor is a static cylinder,
        and the side wall is approximated by static thin boxes around a circle.
        """
        self._ensure_not_finalized()

        if radius <= 0.0:
            raise ValueError("radius must be positive")
        if height <= 0.0:
            raise ValueError("height must be positive")
        if wall_thickness <= 0.0:
            raise ValueError("wall_thickness must be positive")
        if floor_thickness <= 0.0:
            raise ValueError("floor_thickness must be positive")
        if n_faces < 3:
            raise ValueError("n_faces must be at least 3")

        cfg = static_cfg or newton.ModelBuilder.ShapeConfig(
            density=2500.0,
            mu=0.8,
            restitution=0.0,
        )

        cx, cy, cz = center
        static_body = -1

        self.builder.add_ground_plane(
            height=0,
            color=wp.vec3(0.5, 0.5, 0.5),
        )

        # Static shapes use body id -1 in Newton.
        self.builder.add_shape_cylinder(
            static_body,
            radius=radius,
            half_height=floor_thickness * 0.5,
            xform=wp.transform(
                p=wp.vec3(cx, cy, cz + floor_thickness * 0.5),
                q=wp.quat_identity(),
            ),
            cfg=cfg,
            label="container_floor",
        )

        # Each wall segment is tangent to the circle at its center point.
        angle_step = 2.0 * math.pi / n_faces
        wall_radius = radius + wall_thickness * 0.5
        segment_length = 2.0 * wall_radius * math.tan(math.pi / n_faces)

        for i in range(n_faces):
            theta = i * angle_step

            x = cx + wall_radius * math.cos(theta)
            y = cy + wall_radius * math.sin(theta)

            # Box local x points radially, local y follows tangent.
            q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), theta)

            self.builder.add_shape_box(
                static_body,
                hx=wall_thickness * 0.5,
                hy=segment_length * 0.5,
                hz=height * 0.5,
                xform=wp.transform(
                    p=wp.vec3(x, y, cz + height * 0.5),
                    q=q,
                ),
                cfg=cfg,
                label=f"container_wall_{i:03d}",
            )

        self.container_radius = radius
        self.container_height = height


    def add_cuboid(
        self,
        *,
        id: int,
        half_extents: Vec3,
        position: Vec3 = (0.0, 0.0, 0.0),
        quat_xyzw: Vec4 = (0.0, 0.0, 0.0, 1.0),
        density: float = 1200.0,
        mu: float = 0.8,
    ) -> int:
        """Add one dynamic cuboid and return its Newton body id.

        Args:
            id: External object id used later in :meth:`get_simulation_results`.
            half_extents: Box half-size in Newton x, y, z coordinates. For a
                box of size 0.4 x 0.3 x 0.8, use (0.2, 0.15, 0.4).
            position: Initial body position in Newton x, y, z coordinates.
            quat_xyzw: Initial orientation quaternion in x, y, z, w order.
            density: Shape density used by Newton to compute mass properties.
            mu: Friction coefficient.
        """
        self._ensure_not_finalized()

        hx, hy, hz = half_extents
        px, py, pz = position
        qx, qy, qz, qw = quat_xyzw

        cfg = newton.ModelBuilder.ShapeConfig(
            density=density,
            mu=mu,
        )
        body = self.builder.add_body(
            xform=wp.transform(
                p=wp.vec3(px, py, pz),
                q=wp.quat(qx, qy, qz, qw),
            )
        )
        self.builder.add_shape_box(
            body,
            hx=hx,
            hy=hy,
            hz=hz,
            xform=wp.transform_identity(),
            cfg=cfg,
            label=f'cuboid_{id}',
        )

        self._body_id_to_obj_id[body] = id

        return body


    def add_particle(
        self,
        id: int,
        mesh: trimesh.Trimesh,
        position: Vec3 = (0.0, 0.0, 0.0),
        quat_xyzw: Vec4 = (0.0, 0.0, 0.0, 1.0),
        cfg: newton.ModelBuilder.ShapeConfig | None = None,
        build_sdf: bool = True,
    ):
        """Add one dynamic convex mesh particle and return its Newton body id.

        The input mesh is converted to its convex hull before being handed to
        Newton. A signed distance field is built for mesh contacts, so this is
        more expensive than adding a primitive cuboid.

        Args:
            id: External object id used later in :meth:`get_simulation_results`.
            mesh: Prepared particle mesh in Newton x, y, z coordinates.
            position: Initial body position in Newton x, y, z coordinates.
            quat_xyzw: Initial orientation quaternion in x, y, z, w order.
            cfg: Optional collision/material config. A dense, high-friction,
                non-bouncy default is used when omitted.
        """
        self._ensure_not_finalized()

        px, py, pz = position
        qx, qy, qz, qw = quat_xyzw

        body = self.builder.add_body(
            xform=wp.transform(
                p=wp.vec3(px, py, pz),
                q=wp.quat(qx, qy, qz, qw),
            )
        )
        # Newton mesh shapes expect flat int32 triangle indices.
        newton_mesh = newton.Mesh(
            mesh.convex_hull.vertices,
            mesh.convex_hull.faces.reshape(-1).astype(np.int32),
            compute_inertia=False,
            is_solid=True,
        )
        if build_sdf:
            newton_mesh.build_sdf(max_resolution=256, cache_dir='.cache/newton-physics/sdf')
        self.builder.add_shape_mesh(
            body,
            mesh=newton_mesh,
            cfg=cfg or newton.ModelBuilder.ShapeConfig(
                margin=0.0,
                density=1500.0,
                mu=0.8,
                restitution=0.0,
                gap=0.05
            ),
            label=f'particle_{id}',
        )
        self._body_id_to_obj_id[body] = id

        return body


    def finalize(
        self,
        *,
        solver: str = "XPBD",
        solver_iterations: int = 8,
        rigid_contact_relaxation: float = 0.8,
        angular_damping: float = 0.01,
        **kwargs,
    ):
        """Finalize the model and construct the selected Newton solver.

        Args:
            solver: Solver backend. Supported values are ``"XPBD"`` and
                ``"MuJoCo"``.
            solver_iterations: Iterations per solver step.
            rigid_contact_relaxation: XPBD rigid contact relaxation parameter.
            angular_damping: XPBD angular damping applied to rigid bodies.
            njmax: MuJoCo maximum joint constraint count.
            nconmax: MuJoCo maximum contact count.
            ccd_iterations: MuJoCo continuous collision detection iterations.
        """
        self._ensure_not_finalized()
        if solver == "VBD":
            self.builder.color()

        # Only use below code when adding meshes directly
        # self.builder.approximate_meshes(keep_visual_shapes=True)

        self.model = self.builder.finalize(device=self.device)
        if self.viewer is not None:
            self.viewer.set_model(self.model)

        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        if solver == "XPBD":
            self.solver = newton.solvers.SolverXPBD(
                self.model,
                iterations=solver_iterations,
                rigid_contact_relaxation=rigid_contact_relaxation,
                angular_damping=angular_damping,
            )
        elif solver == "MuJoCo":
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                iterations=solver_iterations,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported solver: {solver}")

        self._finalized = True


    def simulate(
        self,
        *,
        fps: int = 60,
        sim_substeps: int = 8,
        num_frames: int = 360,
        progress_bar: bool = True,
    ):
        """Advance the finalized model and return the last simulation state.

        Args:
            fps: Output/render frame rate.
            sim_substeps: Number of smaller physics steps per rendered frame.
                More substeps usually reduce tunnelling and explosive contacts.
            num_frames: Number of rendered frames to simulate.
            progress_bar: Whether to show a ``tqdm`` progress bar.
        """
        if not self._finalized:
            raise RuntimeError("Call finalize() before simulate().")

        frame_dt = 1.0 / fps
        sim_dt = frame_dt / sim_substeps
        sim_time = 0.0

        pbar = range(num_frames)
        if progress_bar:
            pbar = tqdm(pbar, desc="Simulating")

        for _ in pbar:
            for _ in range(sim_substeps):
                self.state_0.clear_forces()

                self.limit_body_speed(self.state_0, max_linear_speed=30, max_angular_speed=10)

                self.model.collide(self.state_0, self.contacts)
                self.solver.step(
                    self.state_0,
                    self.state_1,
                    self.control,
                    self.contacts,
                    sim_dt
                )
                # Newton solvers write into state_1; swap buffers for the next step.
                self.state_0, self.state_1 = self.state_1, self.state_0
            self._log_viewer_frame(sim_time)
            sim_time += frame_dt

        return self.state_0


    def get_simulation_results(self) -> dict[int, SimulationResult]:
        """Return final poses keyed by external object id."""
        if self.state_0 is None:
            raise RuntimeError("Simulation has not been run yet.")
        body_q = self.state_0.body_q.numpy()
        results = {}
        for body_id, obj_id in self._body_id_to_obj_id.items():
            translation = tuple(body_q[body_id, 0:3])
            rotation = tuple(body_q[body_id, 3:7])  # xyzw
            results[obj_id] = SimulationResult(
                obj_id=obj_id,
                body_id=body_id,
                translation_xyz=translation,
                rotation_xyzw=rotation
            )
        return results


    def _log_viewer_frame(self, sim_time) -> None:
        """Send the current state to the optional viewer if it supports logging."""
        if self.viewer is None:
            return
        if hasattr(self.viewer, "begin_frame"):
            self.viewer.begin_frame(sim_time)
        if hasattr(self.viewer, "log_state"):
            self.viewer.log_state(self.state_0)
        if hasattr(self.viewer, "log_contacts"):
            self.viewer.log_contacts(self.contacts, self.state_0)
        if hasattr(self.viewer, "end_frame"):
            self.viewer.end_frame()


    @staticmethod
    def _prepare_mesh(
        mesh: trimesh.Trimesh,
        *,
        scale: float,
        center_mesh: bool,
    ) -> trimesh.Trimesh:
        """Copy, optionally center, reorder, and scale a mesh for Newton.

        Source object volumes use z, y, x indexing, while Newton expects x, y, z
        coordinates. Reordering vertices here keeps later simulation code in
        Newton's coordinate convention.
        """
        prepared = mesh.copy()
        vertices = np.asarray(prepared.vertices, dtype=np.float64).copy()
        if center_mesh:
            vertices -= prepared.bounds.mean(axis=0)
        vertices = vertices[:, [2, 1, 0]]
        prepared.vertices = vertices
        prepared.apply_scale(scale)
        prepared.remove_unreferenced_vertices()
        return prepared


    def limit_body_speed(
        self,
        state,
        *,
        max_linear_speed: float = 2.0,
        max_angular_speed: float = 5.0,
        body_ids: list[int] | None = None,
    ):
        """
        Limit each rigid body's linear and angular speed.

        body_q format:
            [x, y, z, qx, qy, qz, qw]

        body_qd format:
            [vx, vy, vz, wx, wy, wz]

        max_linear_speed:
            Maximum magnitude of [vx, vy, vz].

        max_angular_speed:
            Maximum magnitude of [wx, wy, wz], usually in rad/s.

        height_threshold:
            If given, only bodies with z < height_threshold are limited.

        body_ids:
            If given, only these body IDs are limited.
            Otherwise, all bodies are checked.
        """

        if max_linear_speed < 0:
            raise ValueError("max_linear_speed must be non-negative")
        if max_angular_speed < 0:
            raise ValueError("max_angular_speed must be non-negative")

        body_q = state.body_q.numpy()
        body_qd = state.body_qd.numpy()

        if body_ids is None:
            body_ids = range(body_q.shape[0])

        eps = 1e-8

        for body_id in body_ids:
            # Linear velocity: [vx, vy, vz]
            v = body_qd[body_id, 0:3]
            linear_speed = np.linalg.norm(v)

            if linear_speed > max_linear_speed:
                body_qd[body_id, 0:3] = v * (max_linear_speed / (linear_speed + eps))

            # Angular velocity: [wx, wy, wz]
            w = body_qd[body_id, 3:6]
            angular_speed = np.linalg.norm(w)

            if angular_speed > max_angular_speed:
                body_qd[body_id, 3:6] = w * (max_angular_speed / (angular_speed + eps))

        state.body_qd.assign(body_qd)


    @staticmethod
    def limit_body_acceleration(
        state,
        *,
        max_linear_accel: float = 10.0,
        max_angular_accel: float = 20.0,
        body_ids: list[int] | None = None,
    ):
        """
        Limit each rigid body's linear and angular acceleration by clamping
        the velocity delta between state_prev.body_qd and state.body_qd.

        body_qd format:
            [vx, vy, vz, wx, wy, wz]

        max_linear_accel:
            Maximum magnitude of linear acceleration [m/s^2].

        max_angular_accel:
            Maximum magnitude of angular acceleration [rad/s^2].
        """

        if max_linear_accel < 0:
            raise ValueError("max_linear_speed must be non-negative")
        if max_angular_accel < 0:
            raise ValueError("max_angular_speed must be non-negative")

        body_qdd = state.body_qdd.numpy()

        if body_ids is None:
            body_ids = range(body_qdd.shape[0])

        eps = 1e-8

        for body_id in body_ids:
            # Linear velocity: [vx, vy, vz]
            v = body_qdd[body_id, 0:3]
            linear_speed = np.linalg.norm(v)

            if linear_speed > max_linear_accel:
                body_qdd[body_id, 0:3] = v * (max_linear_accel / (linear_speed + eps))

            # Angular velocity: [wx, wy, wz]
            w = body_qdd[body_id, 3:6]
            angular_speed = np.linalg.norm(w)

            if angular_speed > max_angular_accel:
                body_qdd[body_id, 3:6] = w * (max_angular_accel / (angular_speed + eps))

        state.body_qdd.assign(body_qdd)


    @staticmethod
    def save_simulation_results(
        results: dict[int, SimulationResult],
        output_path: str | Path,
        metadata: dict | None = None,
    ) -> None:
        """Save simulation results and optional metadata to a JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        serializable_results = {
            int(obj_id): result.to_dict()
            for obj_id, result in results.items()
        }
        payload = {
            "metadata": metadata or {},
            "results": serializable_results,
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)


    @staticmethod
    def load_simulation_results(
        input_path: str | Path,
    ) -> tuple[dict[int, "SimulationResult"], dict]:
        input_path = Path(input_path)
        with input_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        metadata = payload.get("metadata", {})
        results = {
            int(obj_id): SimulationResult.from_dict(result_data)
            for obj_id, result_data in payload["results"].items()
        }
        return results, metadata


    @staticmethod
    def verify_simulation_correctness(
        results: dict[int, SimulationResult],
        container_radius: float,
        verbose: bool = True,
    ) -> bool:
        """
        Verify that all simulated objects are within the container radius
        and have non-negative z locations.

        Args:
            results: Mapping from object ID to SimulationResult.
            container_radius: Maximum allowed radial distance from the origin
                in the x-y plane.

        Returns:
            True if all objects are valid, False otherwise.
        """
        is_simulation_correct = True
        for result in tqdm(results.values(), desc="Verifying simulation results"):
            x, y, z = result.translation_xyz
            radial_distance = math.sqrt(x**2 + y**2)
            if radial_distance > container_radius:
                is_simulation_correct = False
                if verbose:
                    print(f"Object {result.obj_id} is outside the container radius: {radial_distance:.2f} > {container_radius:.2f}")
            if z < 0:
                is_simulation_correct = False
                if verbose:
                    print(f"Object {result.obj_id} has negative z location: {z:.2f}")
        return is_simulation_correct
