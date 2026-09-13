#pragma once

#include "mathbr/ols.hpp"

namespace mathbr {

class WLS : public OLS {
public:
    explicit WLS(int n_features) : OLS(n_features) {}
    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y,
             const std::vector<double>& weights) {
        fit_weighted(X, y, weights);
    }
};

}  // namespace mathbr
