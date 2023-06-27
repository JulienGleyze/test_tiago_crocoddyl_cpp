
#include "crocoddyl/core/activations/quadratic-barrier.hpp"
#include "crocoddyl/core/activations/weighted-quadratic.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/core/integrator/euler.hpp"
#include "crocoddyl/core/mathbase.hpp"
#include "crocoddyl/core/optctrl/shooting.hpp"
#include "crocoddyl/core/residuals/control.hpp"
#include "crocoddyl/core/solvers/fddp.hpp"
#include "crocoddyl/core/utils/callbacks.hpp"
#include "crocoddyl/multibody/actions/contact-fwddyn.hpp"
#include "crocoddyl/multibody/actuations/full.hpp"
#include "crocoddyl/multibody/contacts/contact-6d.hpp"
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"
#include "crocoddyl/multibody/residuals/frame-placement.hpp"
#include "crocoddyl/multibody/residuals/state.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "pinocchio//spatial/fwd.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/model.hpp"
#include "pinocchio/fwd.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/spatial/se3-tpl.hpp"

int main(int argc, char** argv) {
  using namespace pinocchio;
  using namespace crocoddyl;

  const std::string urdf_filename = std::string(
      "/opt/openrobots/include/example-robot-data/robots/"
      "tiago_description/robots/tiago.urdf");

  // Load the urdf model
  Model model;
  pinocchio::urdf::buildModel(urdf_filename, model);

  // Create a list of joints to lock
  std::vector<std::string> jointsToLock = {"wheel_left_joint",
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
                                           "hand_thumb_joint"};

  std::vector<FrameIndex> jointsToLockIDs = {};

  for (std::string jn : jointsToLock) {
    jointsToLockIDs.push_back(model.getJointId(jn));
  }

  std::cout << "jointsToLockIDs size: " << jointsToLockIDs.size() << std::endl;

  Eigen::VectorXd q_rand = randomConfiguration(model);

  Model rmodel = buildReducedModel(model, jointsToLockIDs, q_rand);

  std::cout << "Reduced model: " << rmodel << std::endl;

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(rmodel.nq + rmodel.nv);

  // Declaring the foot and hand names
  std::string rw_name = "wheel_right_link";
  std::string lw_name = "wheel_left_link";
  std::vector<std::string> caster_contact_names = {
      "caster_front_right_1_link", "caster_front_right_2_link",
      "caster_front_left_1_link", "caster_front_left_2_link"};
  std::string lh_name = "hand_tool_joint";

  // Getting the frame ids
  FrameIndex rw_id = rmodel.getFrameId(rw_name);
  FrameIndex lw_id = rmodel.getFrameId(lw_name);
  FrameIndex lh_id = rmodel.getFrameId(lh_name);

  std::vector<FrameIndex> caster_contact_ids = {};

  for (std::string caster_contact_name : caster_contact_names) {
    caster_contact_ids.push_back(rmodel.getFrameId(caster_contact_name));
  }

  boost::shared_ptr<Model> shrd_rmodel = boost::make_shared<Model>(rmodel);

  // Define the robot"s state and actuation

  StateMultibody state(shrd_rmodel);

  boost::shared_ptr<StateMultibody> shrd_state =
      boost::make_shared<StateMultibody>(state);

  // actuation = crocoddyl.ActuationModelFloatingBase(state)
  ActuationModelFull actuation(shrd_state);

  actuation.print(std::cout);

  boost::shared_ptr<ActuationModelFull> shrd_actuation =
      boost::make_shared<ActuationModelFull>(actuation);

  const size_t state_nu = actuation.get_nu();

  // Creating a double-support contact (feet support)
  ContactModelMultiple contacts(shrd_state, state_nu);

  boost::shared_ptr<ContactModelMultiple> shrd_contacts =
      boost::make_shared<ContactModelMultiple>(contacts);

  std::vector<ContactModel6D> caster_contacts = {};

  for (FrameIndex caster_contact_id : caster_contact_ids) {
    caster_contacts.push_back(ContactModel6D(shrd_state, caster_contact_id,
                                             SE3::Identity(), state_nu));
  }

  // Define the cost sum (cost manager)
  CostModelSum costs(shrd_state, state_nu);

  boost::shared_ptr<CostModelSum> shrd_costs =
      boost::make_shared<CostModelSum>(costs);

  // Adding the hand-placement cost
  Eigen::VectorXd w_hand(6);

  w_hand << Eigen::VectorXd::Constant(3, 1),
      Eigen::VectorXd::Constant(3, 0.0001);

  Eigen::Vector3d target(0.6, -0.3, 1.5);

  std::cout << "target: " << target.transpose() << std::endl;

  SE3 lh_Mref(Eigen::Matrix3d::Identity(), target);

  ActivationModelWeightedQuad activation_hand(w_hand.cwiseAbs2());

  boost::shared_ptr<ActivationModelWeightedQuad> shrd_act_hand =
      boost::make_shared<ActivationModelWeightedQuad>(activation_hand);

  ResidualModelFramePlacement residual_model_frame_placement(shrd_state, lh_id,
                                                             lh_Mref, state_nu);

  boost::shared_ptr<ResidualModelFramePlacement> shrd_res_mod_frm_plmt =
      boost::make_shared<ResidualModelFramePlacement>(
          residual_model_frame_placement);

  CostModelResidual lh_cost(shrd_state, shrd_act_hand, shrd_res_mod_frm_plmt);

  boost::shared_ptr<CostModelResidual> shrd_lh_cost =
      boost::make_shared<CostModelResidual>(lh_cost);

  costs.addCost("lh_goal", shrd_lh_cost, 1e2);

  const size_t state_nv = state.get_nv();

  // Adding state and control regularization terms
  Eigen::VectorXd w_x(2 * state_nv);
  w_x << Eigen::VectorXd::Zero(3), Eigen::VectorXd::Constant(3, 10.0),
      Eigen::VectorXd::Constant(state_nv - 6, 0.01),
      Eigen::VectorXd::Constant(state_nv, 10.0);

  ActivationModelWeightedQuad activation_xreg(w_x.cwiseAbs2());

  boost::shared_ptr<ActivationModelWeightedQuad> shrd_act_xreg =
      boost::make_shared<ActivationModelWeightedQuad>(activation_xreg);

  ResidualModelState residual_model_state_xreg(shrd_state, x0, state_nu);

  boost::shared_ptr<ResidualModelState> shrd_res_mod_state_xreg =
      boost::make_shared<ResidualModelState>(residual_model_state_xreg);

  ResidualModelControl residual_model_control(shrd_state, state_nu);

  boost::shared_ptr<ResidualModelControl> shrd_res_mod_ctrl =
      boost::make_shared<ResidualModelControl>(residual_model_control);

  CostModelResidual x_reg_cost(shrd_state, shrd_act_xreg,
                               shrd_res_mod_state_xreg);

  CostModelResidual u_reg_cost(shrd_state, shrd_res_mod_ctrl);

  boost::shared_ptr<CostModelResidual> shrd_x_reg_cost =
      boost::make_shared<CostModelResidual>(x_reg_cost);

  boost::shared_ptr<CostModelResidual> shrd_u_reg_cost =
      boost::make_shared<CostModelResidual>(u_reg_cost);

  costs.addCost("xReg", shrd_x_reg_cost, 1e-3);
  costs.addCost("uReg", shrd_u_reg_cost, 1e-4);

  // Adding the state limits penalization
  Eigen::VectorXd x_lb(2 * state_nv);
  x_lb << state.get_lb().head(state_nv), state.get_lb().tail(state_nv);
  Eigen::VectorXd x_ub(2 * state_nv);
  x_ub << state.get_ub().head(state_nv), state.get_ub().tail(state_nv);

  ActivationModelQuadraticBarrier activation_xbounds(
      ActivationBounds(x_lb, x_ub));

  boost::shared_ptr<ActivationModelQuadraticBarrier> shrd_act_xbounds =
      boost::make_shared<ActivationModelQuadraticBarrier>(activation_xbounds);

  ResidualModelState residual_model_state_xbounds(shrd_state, 0 * x0, state_nu);

  boost::shared_ptr<ResidualModelState> shrd_res_mod_state_xbounds =
      boost::make_shared<ResidualModelState>(residual_model_state_xbounds);

  CostModelResidual x_bounds(shrd_state, shrd_act_xbounds,
                             shrd_res_mod_state_xbounds);

  boost::shared_ptr<CostModelResidual> shrd_x_bounds =
      boost::make_shared<CostModelResidual>(x_bounds);

  costs.addCost("xBounds", shrd_x_bounds, 1.0);

  std::cout << "costs: " << costs << std::endl;

  // Creating the action rmodel
  DifferentialActionModelContactFwdDynamics dmodel(shrd_state, shrd_actuation,
                                                   shrd_contacts, shrd_costs);

  std::cout << "Action model: " << dmodel << std::endl;

  boost::shared_ptr<DifferentialActionModelContactFwdDynamics> shrd_dmodel =
      boost::make_shared<DifferentialActionModelContactFwdDynamics>(dmodel);

  double DT = 5e-2;
  const int N = 20;

  // Creating a running rmodel for the target
  std::vector<IntegratedActionModelEuler> running_seqs(
      N, IntegratedActionModelEuler(shrd_dmodel, DT));

  running_seqs.push_back(IntegratedActionModelEuler(shrd_dmodel, 0.0));

  boost::shared_ptr<IntegratedActionModelEuler> shrd_action_model =
      boost::make_shared<IntegratedActionModelEuler>(
          IntegratedActionModelEuler(shrd_dmodel, DT));

  std::vector<boost::shared_ptr<ActionModelAbstract>> shrd_running_seqs(
      N, shrd_action_model);

  IntegratedActionModelEuler mterm =
      IntegratedActionModelEuler(shrd_dmodel, 0.0);

  boost::shared_ptr<IntegratedActionModelEuler> shrd_mterm =
      boost::make_shared<IntegratedActionModelEuler>(mterm);

  // Creating the shooting problem and the FDDP solver

  ShootingProblem problem(x0, shrd_running_seqs, shrd_mterm);

  std::cout << "Problem: " << problem << std::endl;

  std::cout << "Initial state: " << problem.get_x0().transpose() << std::endl;

  boost::shared_ptr<ShootingProblem> shrd_problem =
      boost::make_shared<ShootingProblem>(problem);

  SolverFDDP fddp(shrd_problem);

  // Adding callbacks to inspect the evolution of the solver (logs are
  //     printed in the terminal)
  CallbackVerbose callback_verbose;
  std::vector<boost::shared_ptr<CallbackAbstract>> shrd_callbacks = {
      boost::make_shared<CallbackVerbose>(callback_verbose)};

  fddp.setCallbacks(shrd_callbacks);

  std::cout << "Problem solved:" << std::endl << fddp.solve() << std::endl;
  std::cout << "Number of iterations :" << fddp.get_iter() << std::endl;
  std::cout << "Total cost :" << fddp.get_cost() << std::endl;
  std::cout << "Gradient norm :" << fddp.stoppingCriteria() << std::endl;
}
