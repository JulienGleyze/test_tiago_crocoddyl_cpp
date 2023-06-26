import crocoddyl
import example_robot_data
import numpy as np
import pinocchio as pin

# Load robot
robot = example_robot_data.load("tiago")

# Create a list of joints to lock
jointsToLock = [
    "wheel_left_joint",
    "wheel_right_joint",
    "hand_index_abd_joint",
    "hand_index_virtual_1_joint",
    "hand_index_flex_1_joint",
    "hand_index_virtual_2_joint",
    "hand_index_flex_2_joint",
    "hand_index_virtual_3_joint",
    "hand_index_flex_3_joint",
    "hand_index_joint",
    "hand_little_abd_joint",
    "hand_little_virtual_1_joint",
    "hand_little_flex_1_joint",
    "hand_little_virtual_2_joint",
    "hand_little_flex_2_joint",
    "hand_little_virtual_3_joint",
    "hand_little_flex_3_joint",
    "hand_middle_abd_joint",
    "hand_middle_virtual_1_joint",
    "hand_middle_flex_1_joint",
    "hand_middle_virtual_2_joint",
    "hand_middle_flex_2_joint",
    "hand_middle_virtual_3_joint",
    "hand_middle_flex_3_joint",
    "hand_mrl_joint",
    "hand_ring_abd_joint",
    "hand_ring_virtual_1_joint",
    "hand_ring_flex_1_joint",
    "hand_ring_virtual_2_joint",
    "hand_ring_flex_2_joint",
    "hand_ring_virtual_3_joint",
    "hand_ring_flex_3_joint",
    "hand_thumb_abd_joint",
    "hand_thumb_virtual_1_joint",
    "hand_thumb_flex_1_joint",
    "hand_thumb_virtual_2_joint",
    "hand_thumb_flex_2_joint",
    "hand_thumb_joint",
]

# Get the ID of all existing joints
jointsToLockIDs = []
for jn in jointsToLock:
    if robot.model.existJointName(jn):
        jointsToLockIDs.append(robot.model.getJointId(jn))
    else:
        print("Warning: joint " + str(jn) + " does not belong to the model!")

reduced_robot = robot.buildReducedRobot(jointsToLockIDs)
rmodel = reduced_robot.model
print(rmodel)
q0 = np.zeros(rmodel.nq)
print(q0)

x0 = np.concatenate([q0, np.zeros(rmodel.nv)])

# Declaring the foot and hand names
rw_name = "wheel_right_link"
lw_name = "wheel_left_link"
caster_contact_names = [
    "caster_front_right_1_link",
    "caster_front_right_2_link",
    "caster_front_left_1_link",
    "caster_front_left_2_link",
]
lh_name = "hand_tool_joint"


# Getting the frame ids
rw_id = rmodel.getFrameId(rw_name)
lw_id = rmodel.getFrameId(lw_name)
lh_id = rmodel.getFrameId(lh_name)
caster_contact_ids = []
for caster_contact_name in caster_contact_names:
    caster_contact_ids.append(rmodel.getFrameId(caster_contact_name))

# Define the robot's state and actuation
state = crocoddyl.StateMultibody(rmodel)
# actuation = crocoddyl.ActuationModelFloatingBase(state)
actuation = crocoddyl.ActuationModelFull(state)
print(actuation.nu)


def createActionModel(target):
    # Creating a double-support contact (feet support)
    contacts = crocoddyl.ContactModelMultiple(state, actuation.nu)
    lw_contact = crocoddyl.ContactModel6D(
        state,
        lw_id,
        pin.SE3.Identity(),
        actuation.nu,
        np.array([0, 0]),
    )
    rw_contact = crocoddyl.ContactModel6D(
        state,
        rw_id,
        pin.SE3.Identity(),
        actuation.nu,
        np.array([0, 0]),
    )
    caster_contacts = []
    idx = 0
    for caster_contact_id in caster_contact_ids:
        caster_contacts.append(
            crocoddyl.ContactModel6D(
                state,
                caster_contact_id,
                pin.SE3.Identity(),
                actuation.nu,
                np.array([0, 0]),
            )
        )
        # contacts.addContact(caster_contact_names[idx]+"_contact",caster_contacts[-1])
        idx = idx + 1

        print("caster_contacts for " + str(caster_contact_id))

    # contacts.addContact("lw_contact", lw_contact)
    # contacts.addContact("rw_contact", rw_contact)

    # Define the cost sum (cost manager)
    costs = crocoddyl.CostModelSum(state, actuation.nu)

    # Adding the hand-placement cost
    w_hand = np.array([1] * 3 + [0.0001] * 3)
    lh_Mref = pin.SE3(np.eye(3), target)
    activation_hand = crocoddyl.ActivationModelWeightedQuad(w_hand**2)
    lh_cost = crocoddyl.CostModelResidual(
        state,
        activation_hand,
        crocoddyl.ResidualModelFramePlacement(state, lh_id, lh_Mref, actuation.nu),
    )
    costs.addCost("lh_goal", lh_cost, 1e2)

    # Adding state and control regularization terms
    w_x = np.array([0] * 3 + [10.0] * 3 + [0.01] * (state.nv - 6) + [10] * state.nv)
    print("STATTTTE NV", state.nv)
    activation_xreg = crocoddyl.ActivationModelWeightedQuad(w_x**2)
    x_reg_cost = crocoddyl.CostModelResidual(
        state, activation_xreg, crocoddyl.ResidualModelState(state, x0, actuation.nu)
    )
    u_reg_cost = crocoddyl.CostModelResidual(
        state, crocoddyl.ResidualModelControl(state, actuation.nu)
    )
    costs.addCost("xReg", x_reg_cost, 1e-3)
    costs.addCost("uReg", u_reg_cost, 1e-4)

    # Adding the state limits penalization
    x_lb = np.concatenate([state.lb[1 : state.nv + 1], state.lb[-state.nv :]])
    x_ub = np.concatenate([state.ub[1 : state.nv + 1], state.ub[-state.nv :]])
    activation_xbounds = crocoddyl.ActivationModelQuadraticBarrier(
        crocoddyl.ActivationBounds(x_lb, x_ub)
    )
    x_bounds = crocoddyl.CostModelResidual(
        state,
        activation_xbounds,
        crocoddyl.ResidualModelState(state, 0 * x0, actuation.nu),
    )
    costs.addCost("xBounds", x_bounds, 1.0)

    # Adding the friction cone penalization
    nsurf, mu = np.identity(3), 0.7
    cone = crocoddyl.FrictionCone(nsurf, mu, 4, False)
    activation_friction = crocoddyl.ActivationModelQuadraticBarrier(
        crocoddyl.ActivationBounds(cone.lb, cone.ub)
    )
    lw_friction = crocoddyl.CostModelResidual(
        state,
        activation_friction,
        crocoddyl.ResidualModelContactFrictionCone(state, lw_id, cone, actuation.nu),
    )
    rw_friction = crocoddyl.CostModelResidual(
        state,
        activation_friction,
        crocoddyl.ResidualModelContactFrictionCone(state, rw_id, cone, actuation.nu),
    )

    idx = 0
    for caster_contact_id in caster_contact_ids:
        caster_friction = crocoddyl.CostModelResidual(
            state,
            activation_friction,
            crocoddyl.ResidualModelContactFrictionCone(
                state, caster_contact_id, cone, actuation.nu
            ),
        )
        # costs.addCost(caster_contact_names[idx]+"_friction",caster_friction,1e1)
        print("caster_contact_id: " + str(caster_contact_id) + " " + caster_contact_names[idx])
        idx = idx + 1

    # costs.addCost("lw_friction", lw_friction, 1e1)
    # costs.addCost("rw_friction", rw_friction, 1e1)

    # Creating the action model
    dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contacts, costs)
    return dmodel


def createSequence(dmodels, DT, N):
    return [
        [crocoddyl.IntegratedActionModelEuler(m, DT)] * N
        + [crocoddyl.IntegratedActionModelEuler(m, 0.0)]
        for m in dmodels
    ]


import meshcat.geometry as g
import meshcat.transformations as tf


def createDisplay(targets):
    display = crocoddyl.MeshcatDisplay(reduced_robot, 4, 4, False)
    for i, target in enumerate(targets):
        display.robot.viewer["target_" + str(i)].set_object(g.Sphere(0.05))
        Href = np.array(
            [
                [1.0, 0.0, 0.0, target[0]],
                [0.0, 1.0, 0.0, target[1]],
                [0.0, 0.0, 1.0, target[2]],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        display.robot.viewer["target_" + str(i)].set_transform(
            np.array(
                [
                    [1.0, 0.0, 0.0, target[0]],
                    [0.0, 1.0, 0.0, target[1]],
                    [0.0, 0.0, 1.0, target[2]],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )
    return display


DT, N = 5e-2, 20
target = np.array([0.6, -0.3, 1.5])

# Creating a running model for the target
dmodel = createActionModel(target)
seqs = createSequence([dmodel], DT, N)

# Defining the problem and the solver
problem = crocoddyl.ShootingProblem(x0, sum(seqs, [])[:-1], seqs[-1][-1])
print("\n \n")
print(sum(seqs, [])[:-1])
print("\n \n")
print(seqs)
fddp = crocoddyl.SolverFDDP(problem)
print(fddp)

# Creating display
display = createDisplay([target])

# Adding callbacks to inspect the evolution of the solver (logs are printed in the terminal)
fddp.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])

# Embedded in this cell
# display.robot.viewer.jupyter_cell()

print("Problem solved:", fddp.solve())
print("Number of iterations:", fddp.iter)
print("Total cost:", fddp.cost)
print("Gradient norm:", fddp.stoppingCriteria())

display.rate = -1
display.freq = 1
display.displayFromSolver(fddp)
