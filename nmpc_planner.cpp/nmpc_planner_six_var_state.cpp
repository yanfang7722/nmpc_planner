#include "nmpc_planner_six_var_state.h"

namespace nio {

NmpcPlanner6Var::NmpcPlanner6Var(const std::vector<State>& path, const State& state,
                                 const NmpcConfig& config)
    : NmpcPlanner(path, state, config) {
  C_x_i_.rightCols(num_cx_) = Eigen::MatrixXd::Identity(num_cx_, num_cx_);
  initWieght();
}

void NmpcPlanner6Var::initWieght() {
  for (int refer_idx = 0; refer_idx < refer_path_.size()-1; refer_idx++) {
    x_refer_.segment(refer_idx * num_x_, num_x_) = StateToXVector(refer_path_[refer_idx+1]);
    u_refer_.segment(refer_idx * num_u_, num_u_) = StateToUVector(refer_path_[refer_idx]);
  }
  u_result_ = u_refer_;
  x_init_ = StateToXVector(current_state_);
  std::cout << "x_refer_:\n" << x_refer_ << std::endl;
  std::cout << "u_refer_:\n" << u_refer_ << std::endl;
  std::cout << "x_init_:\n" << x_init_ << std::endl;

  Q_i_ = 100 * Eigen::MatrixXd::Identity(num_x_, num_x_);
  Q_i_(xst::theta, xst::theta) = 2;
  Q_i_(xst::v,xst::v) = 2;
  R_i_ = Eigen::MatrixXd::Identity(num_u_, num_u_);
  for (int refer_idx = 0; refer_idx < refer_path_.size()-1; refer_idx++) {
    Q_.block(refer_idx * num_x_, refer_idx * num_x_, num_x_, num_x_) = Q_i_;
    R_.block(refer_idx * num_u_, refer_idx * num_u_, num_u_, num_u_) = R_i_;
  }
  u_result_ = u_refer_;
  std::cout << "Q_i:\n" << Q_i_ << std::endl;
  std::cout << "R_i:\n" << R_i_ << std::endl;
}

Eigen::VectorXd NmpcPlanner6Var::StateToXVector(const State& st) const {
  Eigen::VectorXd x = Eigen::VectorXd::Zero(num_x_);
  x(xst::x) = st.x;
  x(xst::y) = st.y;
  x(xst::theta) = st.theta;
  x(xst::v) = st.v;
  x(xst::a) = st.acc;
  x(xst::w) = st.w;
  return x;
}

Eigen::VectorXd NmpcPlanner6Var::StateToUVector(const State& st) const {
  Eigen::VectorXd u = Eigen::VectorXd::Zero(num_u_);
  u(ust::jerk) = st.jerk;
  u(ust::dw) = st.dw;
  return u;
}

// void NmpcPlanner6Var::updateCurrentState() {
//   u_result_ = optimize_res_ + u_result_;
//   std::cout << "u_real:\n" << u_result_ << std::endl;
//   Eigen::VectorXd x_update = x_init_;
//   for (int refer_idx = 0; refer_idx < refer_path_.size()-1; refer_idx++) {
//     Eigen::VectorXd u = u_result_.segment(refer_idx * num_u_, num_u_);
//     x_update = EulerSolve(x_update, u, dt_);
//     x_result_.segment(refer_idx * num_x_, num_x_) = x_update;
//   }
//   std::cout << "x_result_:" << x_result_.transpose() << std::endl;
// }

void NmpcPlanner6Var::updateStateFunction(const int predict_idx) {
  A_i_ = Eigen::MatrixXd::Identity(num_x_, num_x_);
  B_i_ = Eigen::MatrixXd::Zero(num_x_, num_u_);
  // C_i_ = Eigen::MatrixXd::Zero(num_c_, num_x_);

  double dt = dt_;
  double theta = x_result_(predict_idx * num_x_ + xst::theta);
  double v = x_result_(predict_idx * num_x_ + xst::v);
  double rho = tan(x_result_(predict_idx * num_x_ + xst::w));

  A_i_(xst::x, xst::theta) = -sin(theta) * dt * v;
  A_i_(xst::x, xst::v) = dt * cos(theta);
  A_i_(xst::y, xst::theta) = cos(theta) * dt * v;
  A_i_(xst::y, xst::v) = dt * sin(theta);
  A_i_(xst::theta, xst::v) = rho / wheel_base * dt;
  A_i_(xst::theta, xst::w) = v * (rho * rho + 1) / wheel_base * dt;
  A_i_(xst::v, xst::a) = dt;
  B_i_(xst::a, ust::jerk) = dt;
  B_i_(xst::w, ust::dw) = dt;
  // std::cout << "B_i_:\n" << B_i_ << std::endl;

  // C_i_.rightCols(num_u_) = Eigen::Matrix2d::Identity();
  // std::cout << "C_i_:\n" << C_i_ << std::endl;
}

void NmpcPlanner6Var::updatePointConstrain(const int i) {
  qp_lb_(i * num_c_) = cons_lower_(0) - x_result_(i * num_x_ + xst::v);
  qp_ub_(i * num_c_) = cons_upper_(0) - x_result_(i * num_x_ + xst::v);
  // acc constrain: a_ = a - v_result
  qp_lb_(i * num_c_ + 1) = cons_lower_(1) - x_result_(i * num_x_ + xst::a);
  qp_ub_(i * num_c_ + 1) = cons_upper_(1) - x_result_(i * num_x_ + xst::a);
  // front_angle: w_= w - w_result
  qp_lb_(i * num_c_ + 2) = cons_lower_(2) - x_result_(i * num_x_ + xst::w);
  qp_ub_(i * num_c_ + 2) = cons_upper_(2) - x_result_(i * num_x_ + xst::w);
  qp_a_.middleRows(i * num_c_, num_cx_) = C_x_i_ * B_integ_i_;

  qp_lb_.segment(i * num_c_ + num_cx_, num_u_) =
      cons_lower_.bottomRows(num_u_) - u_result_.segment(i * num_u_, num_u_);
  qp_ub_.segment(i * num_c_ + num_cx_, num_u_) =
      cons_upper_.bottomRows(num_u_) - u_result_.segment(i * num_u_, num_u_);
  qp_a_.block(i * num_c_ + num_cx_, i * num_u_, num_u_, num_u_) = Eigen::Matrix2d::Identity();
}

// void NmpcPlanner6Var::updateStateMatrix() {
//   A_integ_i_ = Eigen::MatrixXd::Identity(num_x_, num_x_);
//   B_integ_i_ = Eigen::MatrixXd::Zero(num_x_, num_u_ * num_r_);

//   for (int i = 0; i < num_r_; ++i) {
//     updateStateFunction(i);
//     // TODO::
//     A_integ_i_ = A_i_ * A_integ_i_;
//     A_.middleRows(i * num_x_, num_x_) = A_integ_i_;
//     // std::cout << "A_integ_i_:\n" << A_integ_i_ << std::endl;

//     B_integ_i_ = A_i_ * B_integ_i_;
//     B_integ_i_.middleCols(i * num_u_, num_u_) = B_i_;
//     B_.middleRows(i * num_x_, num_x_) = B_integ_i_;
//     // std::cout << "B_integ_i_:\n" << B_integ_i_.leftCols((i + 1) * num_u_) << std::endl;
//     // std::cout << "B_integ_i_:\n" << B_integ_i_ << std::endl;
//     updatePointConstrain(i);
//     // std::cout << "qp_a_.block:\n" << qp_a_ << std::endl;
//   }

//   // end point constrain

//   hessian_ = 2 * (B_.transpose() * Q_ * B_ + R_);
//   // gradient_ = 2 * (A_ * dx_init_).transpose() * Q_ * B_;
//   gradient_ = 2 * B_.transpose() * Q_ * (x_result_ - x_refer_) + 2 * R_ * u_result_;
//   // std::cout << "B_:\n" << B_ << std::endl;
//   // std::cout << "A_:\n" << A_ << std::endl;
//   // std::cout << "Q_:\n" << Q_ << std::endl;
//   // std::cout << " A_ * dx_init_:\n" << (x_result_ - x_refer_) << std::endl;
//   // std::cout << "hessian_:\n" << hessian_ << std::endl;
//   // std::cout << "gradient_:\n" << gradient_ << std::endl;
//   // std::cout << "qp_a_:\n" << qp_a_ << std::endl;
//   // std::cout << "qp_lb_:\n" << qp_lb_ << std::endl;
//   // std::cout << "qp_ub_:\n" << qp_ub_ << std::endl;
//   // std::cout << "res:\n" << hessian_.inverse() * gradient_ << std::endl;
// }

Eigen::VectorXd NmpcPlanner6Var::GetDeltaVal(const Eigen::VectorXd& x,
                                             const Eigen::VectorXd& u) const {
  Eigen::VectorXd res = Eigen::VectorXd::Zero(num_x_);
  res(xst::x) = cos(x(xst::theta)) * x(xst::v);  // dx=v*cos(theta)
  res(xst::y) = sin(x(xst::theta)) * x(xst::v);  // dy= v*sin(theta)
  res(xst::theta) = tan(x(xst::w)) / wheel_base * x(xst::v);
  res(xst::v) = x(xst::a);
  res(xst::a) = u(ust::jerk);
  res(xst::w) = u(ust::dw);
  return res;
}

std::vector<State> NmpcPlanner6Var::getPredictTrajectory() const {
  std::vector<State> res;
  res.push_back(current_state_);
  for (int i = 0; i < num_r_; i++) {
    auto x_update = x_result_.segment(i * num_x_, num_x_);
    State st;
    st.jerk = u_result_[i * num_u_ + ust::jerk];
    st.dw = u_result_[i * num_u_ + ust::dw];
    st.x = x_update[xst::x];
    st.y = x_update[xst::y];
    st.theta = x_update[xst::theta];
    st.v = x_update[xst::v];
    st.acc = x_update[xst::a];
    st.w = x_update[xst::w];
    st.t = current_state_.t + i * dt_;
    res.push_back(st);
  }
  std::cout << "path size:" << res.size() << std::endl;
  return res;
}

NmpcPlanner6Var::~NmpcPlanner6Var() {}

}  // namespace nio
