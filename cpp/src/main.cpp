
#include "/home/julien/stage_laas_gepetto/test_tiago_crocoddyl_ws/cpp/include/tiago_whole_body_manipulator.hpp"

int main(int argc, char** argv) {
  using namespace pinocchio;
  using namespace crocoddyl;
  using namespace tiago_manipulator;

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
                                           "hand_thumb_joint",
                                           "torso_lift_joint",
                                           "head_1_joint",
                                           "head_2_joint"};

  std::vector<FrameIndex> jointsToLockIDs = {};

  for (std::string jn : jointsToLock) {
    if (model.existJointName(jn)) {
      jointsToLockIDs.push_back(model.getJointId(jn));
    } else {
      std::cout << "Joint " << jn << " not found in the model" << std::endl;
    }
  }

  std::cout << "jointsToLockIDs size: " << jointsToLockIDs.size() << std::endl;

  // Random configuration for the reduced model

  Eigen::VectorXd q_rand = randomConfiguration(model);

  Model rmodel = buildReducedModel(model, jointsToLockIDs, q_rand);

  std::cout << "Reduced model: " << rmodel << std::endl;

  // Initial state for the solver
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(rmodel.nq + rmodel.nv);

  // Declaring the foot and hand names
  std::string rw_name = "wheel_right_link";
  std::string lw_name = "wheel_left_link";
  std::vector<std::string> caster_contact_names = {
      "caster_front_right_1_link", "caster_front_right_2_link",
      "caster_front_left_1_link", "caster_front_left_2_link"};
  std::string lh_name = "hand_tool_joint";
  FrameIndex lh_id = rmodel.getJointId(lh_name);

  Eigen::Vector3d target(0.6, -0.3, 1.5);

  OCP OCP_tiago(rmodel);

  OCP_tiago.setX0(x0);
  OCP_tiago.setLhId(lh_id);
  OCP_tiago.setTarget(target);
  OCP_tiago.setHorizonLength(20);

  OCP_tiago.buildCostsModel();
  OCP_tiago.buildDiffActModel();
  OCP_tiago.buildSolver();

  OCP_tiago.printCosts();
  OCP_tiago.printProblem();

  CallbackVerbose callback_verbose;

  OCP_tiago.createCallbacks(callback_verbose);

  OCP_tiago.solve(Eigen::VectorXd::Zero(rmodel.nq + rmodel.nv));

  OCP_tiago.logSolverData();
}
