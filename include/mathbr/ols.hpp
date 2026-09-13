#pragma once

#include <vector>

namespace mathbr {

class OLS {
public:
    explicit OLS(int n_features);
    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y);
    void fit_weighted(const std::vector<std::vector<double>>& X, const std::vector<double>& y,
                      const std::vector<double>& weights);
    double predict(const std::vector<double>& x) const;
    std::vector<double> predict_batch(const std::vector<std::vector<double>>& X) const;
    std::vector<double> coefficients() const;
    std::vector<double> standard_errors() const;
    std::vector<double> t_statistics() const;
    std::vector<double> p_values() const;
    std::vector<std::vector<double>> confidence_intervals(double level = 0.95) const;
    double r_squared() const;
    double adjusted_r_squared() const;
    double f_statistic() const;
    double residual_variance() const;
    int degrees_of_freedom() const;
    bool trained() const noexcept;

private:
    void require_trained() const;
    int n_features_;
    std::vector<double> coefficients_;
    std::vector<double> standard_errors_;
    double r_squared_ = 0.0;
    double residual_variance_ = 0.0;
    int degrees_of_freedom_ = 0;
    bool trained_ = false;
};

}  // namespace mathbr
