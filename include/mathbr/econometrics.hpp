#pragma once

#include <unordered_map>
#include <vector>

namespace mathbr {

class IV2SLS {
public:
    IV2SLS(int n_exog, int n_endog, int n_instruments);
    void fit(const std::vector<std::vector<double>>& exog,
             const std::vector<std::vector<double>>& endog,
             const std::vector<std::vector<double>>& instruments,
             const std::vector<double>& y);
    double predict(const std::vector<double>& exog, const std::vector<double>& endog) const;
    std::vector<double> coefficients() const;
    std::vector<double> first_stage_r_squared() const;
    bool trained() const noexcept;

private:
    void require_trained() const;
    int n_exog_, n_endog_, n_instruments_;
    std::vector<double> coefficients_, first_stage_r_squared_;
    bool trained_ = false;
};

class FixedEffects {
public:
    explicit FixedEffects(int n_features);
    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<double>& y, const std::vector<int>& entity_ids);
    double predict(const std::vector<double>& x, int entity_id) const;
    std::vector<double> coefficients() const;
    double entity_intercept(int entity_id) const;
    double within_r_squared() const;
    bool trained() const noexcept;

private:
    void require_trained() const;
    int n_features_;
    std::vector<double> coefficients_;
    std::unordered_map<int, double> intercepts_;
    double within_r_squared_ = 0.0;
    bool trained_ = false;
};

}  // namespace mathbr
