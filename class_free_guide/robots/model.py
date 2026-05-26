from pinocchio import casadi as cpin
import pinocchio as pin
import numpy as np


class robot_dynamics:

    def __init__(self, dynamics_name: str, urdf_path: str):
        self._model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self._data = self._model.createData()
        self.q = np.array(self._model.nq)
        self.v = np.array(self._model.nv)
        self.last_v = np.array(self._model.nv)
        self.rot_mat = np.eye(3)
        self.control_dt = 0.02
        self.update_kin = False
        self.M = None
        self.C = None
        self.G = None
        self.CMM = None
        self.dCMM = None
        self.CM = None
        self.dCM = None
        self.CoM = None

    def model(self):
        return self._model

    @property
    def data(self):
        if not self.update_kin:
            self.forwardKinematics()
        return self._data

    def update_state(self, base_pos, base_vel, joint_pos, joint_vel, dt):
        self.rot_mat = pin.Quaternion(base_pos[3:7]).toRotationMatrix()
        self.q = np.concatenate((base_pos, joint_pos))
        self.v = np.concatenate((base_vel, joint_vel))
        self.control_dt = dt
        self.acc = (self.v - self.last_v) / self.control_dt
        self.last_v = self.v.copy()
        self.update_kin = False

    def forwardKinematics(self):
        pin.forwardKinematics(self._model, self._data, self.q, self.v, self.acc)
        pin.updateFramePlacements(self._model, self._data)
        self.update_kin = True

    def WB_dynamics(self):
        if not self.update_kin:
            self.forwardKinematics()
        # 质心动量矩阵(A矩阵)
        self.M = pin.crba(self._model, self._data, self.q)
        self.C = pin.nonLinearEffects(self._model, self._data, self.q, self.v)
        self.G = pin.computeGeneralizedGravity(self._model, self._data, self.q)

    # com dynamic Algorithm
    def COM_dynamic(self):
        if not self.update_kin:
            self.forwardKinematics()
        # 质心动量矩阵(A矩阵)
        self.CMM = pin.computeCentroidalMap(self._model, self._data, self.q)
        self.dCMM = pin.computeCentroidalMapTimeVariation(self._model, self._data, self.q, self.v)
        # 基座系下的质心动量
        self.CM = pin.computeCentroidalMomentum(self._model, self._data, self.q, self.v)
        self.dCM = pin.computeCentroidalMomentumTimeVariation(self._model, self._data, self.q, self.v, self.acc)
        # 世界坐标系下的质心位置
        self.CoM = pin.centerOfMass(self._model, self._data, self.q)

    def dls_ik(self, frame_name, target_pose, q_init=None, max_iter=100, tol=1e-4, damping=1e-3):
        if q_init is None:
            q = self.q.copy()
        else:
            q = q_init.copy()

        frame_id = self._model.getFrameId(frame_name)
        if frame_id == len(self._model.frames):
            raise ValueError(f"Frame '{frame_name}' not found in the model.")

        for _ in range(max_iter):
            pin.forwardKinematics(self._model, self._data, q)
            pin.updateFramePlacements(self._model, self._data)

            current_pose = self._data.oMf[frame_id]
            error_se3 = pin.log(current_pose.inverse() * target_pose).vector
            if np.linalg.norm(error_se3) < tol:
                break

            jacobian = pin.computeFrameJacobian(self._model, self._data, q, frame_id, pin.ReferenceFrame.LOCAL)

            jj_t = jacobian @ jacobian.T
            damped = jj_t + (damping**2) * np.eye(6)
            delta = jacobian.T @ np.linalg.solve(damped, error_se3)
            q = pin.integrate(self._model, q, delta)

        return q
