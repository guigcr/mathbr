#pragma once

#include <vector>

namespace mathbr {

class VAR {
public:
    VAR(int n_series, int lags);
    void fit(const std::vector<std::vector<double>>& observations);
    std::vector<std::vector<double>> coefficients() const;
    std::vector<std::vector<double>> forecast(int steps) const;
    std::vector<std::vector<double>> residual_covariance() const;
    bool trained() const noexcept;

private:
    void require_trained() const;
    int n_series_, lags_;
    std::vector<std::vector<double>> coefficients_, history_, covariance_;
    bool trained_ = false;
};

class AR {
public:
    explicit AR(int lags);
    void fit(const std::vector<double>& observations);
    std::vector<double> coefficients() const;
    std::vector<double> forecast(int steps) const;
    bool trained() const noexcept;

private:
    VAR model_;
};

class GARCH {
public:
    GARCH(int max_iter = 2000, double tol = 1e-7, bool arch_only = false);
    void fit(const std::vector<double>& observations);
    double mean() const;
    double omega() const;
    double alpha() const;
    double beta() const;
    double log_likelihood() const;
    std::vector<double> conditional_variance() const;
    std::vector<double> forecast_variance(int steps) const;
    bool converged() const;
    bool trained() const noexcept;

private:
    void require_trained() const;
    int max_iter_;
    double tol_;
    bool arch_only_;
    double mean_ = 0.0, omega_ = 0.0, alpha_ = 0.0, beta_ = 0.0, log_likelihood_ = 0.0;
    double last_error_squared_ = 0.0;
    std::vector<double> variance_;
    bool converged_ = false, trained_ = false;
};

class ARCH : public GARCH {
public:
    ARCH(int max_iter = 2000, double tol = 1e-7) : GARCH(max_iter, tol, true) {}
};

}  // namespace mathbr
