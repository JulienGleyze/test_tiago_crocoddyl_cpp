#ifndef TIAGO_WHOLE_BODY_MANIPULATOR__TIAGO_WHOLE_BODY_MANIPULATOR_HPP

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

using namespace pinocchio;
using namespace crocoddyl;
using namespace Eigen;

namespace tiago_manipulator {
class OCP {
 private:
  void initOCPParms();

  boost::shared_ptr<ShootingProblem> problem_;

  boost::shared_ptr<SolverFDDP> solver_;
  Model model_;

  boost::shared_ptr<crocoddyl::StateMultibody> state_;
  boost::shared_ptr<crocoddyl::ActuationModelFull> actuation_;
  boost::shared_ptr<CostModelSum> costs_;
  boost::shared_ptr<ContactModelMultiple> contacts_;

  boost::shared_ptr<DifferentialActionModelContactFwdDynamics> diff_act_model_;

  bool is_initialized_ = false;

  size_t horizon_length_ = 20;
  double time_step_ = 5e-2;

  size_t actuation_nu_;
  size_t state_nv_;
  size_t state_nq_;

  Vector3d target_;
  VectorXd x0_;

  SE3 lh_Mref_;

  FrameIndex lh_id_;

 public:
  OCP(const Model &model);

  void buildCostsModel();
  void buildDiffActModel();
  void buildSolver();

  void createCallbacks(CallbackVerbose &callbacks);
  void solve(const VectorXd &measured_x);

  void setTarget(Vector3d &target);
  void setX0(VectorXd &x0);
  void setTimeStep(double time_step);
  void setHorizonLength(int horizon_length);
  void setLhId(FrameIndex &lh_id);

  void printCosts() { std::cout << *costs_ << std::endl; };
  void printProblem() { std::cout << *problem_ << std::endl; };
  void logSolverData();

  const Vector3d get_target();
  const VectorXd get_torque();
  const MatrixXd get_gain();

  boost::shared_ptr<StateMultibody> get_state() { return state_; }
};
};  // namespace tiago_manipulator

#endif
