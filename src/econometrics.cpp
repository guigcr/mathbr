#include "mathbr/econometrics.hpp"
#include "mathbr/ols.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace mathbr {

namespace {
void check_rows(const std::vector<std::vector<double>>& rows, size_t n, size_t columns,
                const char* name) {
    if (rows.size() != n) throw std::invalid_argument("input row counts must match");
    for (const auto& row : rows) {
        if (row.size() != columns) throw std::invalid_argument(name);
        for (double value : row)
            if (!std::isfinite(value)) throw std::invalid_argument("features must be finite");
    }
}
}

IV2SLS::IV2SLS(int n_exog, int n_endog, int n_instruments)
    : n_exog_(n_exog), n_endog_(n_endog), n_instruments_(n_instruments) {
    if (n_exog < 0 || n_endog <= 0 || n_instruments < n_endog)
        throw std::invalid_argument("IV2SLS requires endogenous features and at least as many excluded instruments");
}

void IV2SLS::require_trained() const {
    if (!trained_) throw std::logic_error("fit must be called first");
}

void IV2SLS::fit(const std::vector<std::vector<double>>& exog,
                 const std::vector<std::vector<double>>& endog,
                 const std::vector<std::vector<double>>& instruments,
                 const std::vector<double>& y) {
    const size_t n = y.size();
    if (n <= static_cast<size_t>(1 + n_exog_ + n_instruments_) ||
        n <= static_cast<size_t>(1 + n_exog_ + n_endog_))
        throw std::invalid_argument("too few observations for IV2SLS stages");
    check_rows(exog, n, n_exog_, "exog rows must match n_exog");
    check_rows(endog, n, n_endog_, "endog rows must match n_endog");
    check_rows(instruments, n, n_instruments_, "instrument rows must match n_instruments");
    for (double value : y)
        if (!std::isfinite(value)) throw std::invalid_argument("y must be finite");

    std::vector<std::vector<double>> first_X(n), second_X(n);
    for (size_t i = 0; i < n; ++i) {
        first_X[i] = exog[i];
        first_X[i].insert(first_X[i].end(), instruments[i].begin(), instruments[i].end());
        second_X[i] = exog[i];
    }
    std::vector<double> first_r2;
    for (int j = 0; j < n_endog_; ++j) {
        std::vector<double> target(n);
        for (size_t i = 0; i < n; ++i) target[i] = endog[i][j];
        OLS stage(n_exog_ + n_instruments_);
        stage.fit(first_X, target);
        first_r2.push_back(stage.r_squared());
        auto projected = stage.predict_batch(first_X);
        for (size_t i = 0; i < n; ++i) second_X[i].push_back(projected[i]);
    }
    OLS second(n_exog_ + n_endog_);
    second.fit(second_X, y);
    coefficients_ = second.coefficients();
    first_stage_r_squared_ = std::move(first_r2);
    trained_ = true;
}

double IV2SLS::predict(const std::vector<double>& exog, const std::vector<double>& endog) const {
    require_trained();
    if (exog.size() != static_cast<size_t>(n_exog_) || endog.size() != static_cast<size_t>(n_endog_))
        throw std::invalid_argument("prediction feature counts do not match model");
    double result = coefficients_[0];
    for (int j = 0; j < n_exog_; ++j) result += coefficients_[j + 1] * exog[j];
    for (int j = 0; j < n_endog_; ++j) result += coefficients_[n_exog_ + j + 1] * endog[j];
    return result;
}

std::vector<double> IV2SLS::coefficients() const { require_trained(); return coefficients_; }
std::vector<double> IV2SLS::first_stage_r_squared() const { require_trained(); return first_stage_r_squared_; }
bool IV2SLS::trained() const noexcept { return trained_; }

FixedEffects::FixedEffects(int n_features) : n_features_(n_features) {
    if (n_features <= 0) throw std::invalid_argument("n_features must be positive");
}

void FixedEffects::require_trained() const {
    if (!trained_) throw std::logic_error("fit must be called first");
}

void FixedEffects::fit(const std::vector<std::vector<double>>& X,
                       const std::vector<double>& y, const std::vector<int>& entity_ids) {
    const size_t n = y.size();
    if (n <= static_cast<size_t>(n_features_ + 1) || entity_ids.size() != n)
        throw std::invalid_argument("panel inputs must have matching lengths and enough observations");
    check_rows(X, n, n_features_, "X rows must match n_features");
    struct Group { size_t count = 0; double sum_y = 0.0; std::vector<double> sum_x; };
    std::unordered_map<int, Group> groups;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(y[i])) throw std::invalid_argument("y must be finite");
        auto& group = groups[entity_ids[i]];
        if (group.sum_x.empty()) group.sum_x.resize(n_features_, 0.0);
        ++group.count;
        group.sum_y += y[i];
        for (int j = 0; j < n_features_; ++j) group.sum_x[j] += X[i][j];
    }
    std::vector<std::vector<double>> centered_X(n, std::vector<double>(n_features_));
    std::vector<double> centered_y(n);
    for (size_t i = 0; i < n; ++i) {
        const auto& group = groups.at(entity_ids[i]);
        centered_y[i] = y[i] - group.sum_y / group.count;
        for (int j = 0; j < n_features_; ++j)
            centered_X[i][j] = X[i][j] - group.sum_x[j] / group.count;
    }
    OLS within(n_features_);
    within.fit(centered_X, centered_y);
    const auto fitted = within.coefficients();
    std::vector<double> slopes(fitted.begin() + 1, fitted.end());
    std::unordered_map<int, double> intercepts;
    for (const auto& item : groups) {
        double intercept = item.second.sum_y / item.second.count;
        for (int j = 0; j < n_features_; ++j)
            intercept -= slopes[j] * item.second.sum_x[j] / item.second.count;
        intercepts[item.first] = intercept;
    }
    coefficients_ = std::move(slopes);
    intercepts_ = std::move(intercepts);
    within_r_squared_ = within.r_squared();
    trained_ = true;
}

double FixedEffects::entity_intercept(int entity_id) const {
    require_trained();
    auto item = intercepts_.find(entity_id);
    if (item == intercepts_.end()) throw std::invalid_argument("unknown entity_id");
    return item->second;
}

double FixedEffects::predict(const std::vector<double>& x, int entity_id) const {
    if (x.size() != static_cast<size_t>(n_features_))
        throw std::invalid_argument("x size doesn't match n_features");
    double result = entity_intercept(entity_id);
    for (int j = 0; j < n_features_; ++j) result += coefficients_[j] * x[j];
    return result;
}

std::vector<double> FixedEffects::coefficients() const { require_trained(); return coefficients_; }
double FixedEffects::within_r_squared() const { require_trained(); return within_r_squared_; }
bool FixedEffects::trained() const noexcept { return trained_; }

}  // namespace mathbr
