#pragma once

#include <vector>

namespace mathbr {

class RegularizedRegression {
public:
    RegularizedRegression(int n_features, double alpha, double l1_ratio,
                          int max_iter, double tol);
    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y);
    double predict(const std::vector<double>& x) const;
    std::vector<double> predict_batch(const std::vector<std::vector<double>>& X) const;
    std::vector<double> coefficients() const;
    double intercept() const;
    int iterations() const;
    bool converged() const;
    bool trained() const noexcept;

private:
    void require_trained() const;
    int n_features_;
    double alpha_;
    double l1_ratio_;
    int max_iter_;
    double tol_;
    std::vector<double> coefficients_;
    double intercept_ = 0.0;
    int iterations_ = 0;
    bool converged_ = false;
    bool trained_ = false;
};

class Ridge : public RegularizedRegression {
public:
    Ridge(int n_features, double alpha = 1.0, int max_iter = 1000, double tol = 1e-8)
        : RegularizedRegression(n_features, alpha, 0.0, max_iter, tol) {}
};

class Lasso : public RegularizedRegression {
public:
    Lasso(int n_features, double alpha = 1.0, int max_iter = 1000, double tol = 1e-8)
        : RegularizedRegression(n_features, alpha, 1.0, max_iter, tol) {}
};

class ElasticNet : public RegularizedRegression {
public:
    ElasticNet(int n_features, double alpha = 1.0, double l1_ratio = 0.5,
               int max_iter = 1000, double tol = 1e-8)
        : RegularizedRegression(n_features, alpha, l1_ratio, max_iter, tol) {}
};

}  // namespace mathbr
