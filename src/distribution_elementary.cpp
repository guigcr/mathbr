#include "mathbr/distributions.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace mathbr::distributions {
namespace {
constexpr double pi = 3.1415926535897932384626433832795;
void finite(double x) {
    if (!std::isfinite(x)) throw std::invalid_argument("arguments must be finite");
}
void positive(double x) {
    finite(x);
    if (x <= 0) throw std::invalid_argument("scale, rate, and shape must be positive");
}
void probability(double p) {
    finite(p);
    if (p < 0 || p > 1) throw std::invalid_argument("p must be in [0, 1]");
}
}  // namespace

double exponential_pdf(double x, double rate) {
    finite(x); positive(rate);
    return x < 0 ? 0 : rate * std::exp(-rate * x);
}
double exponential_cdf(double x, double rate) {
    finite(x); positive(rate);
    return x < 0 ? 0 : -std::expm1(-rate * x);
}
double exponential_ppf(double p, double rate) {
    probability(p); positive(rate);
    return p == 1 ? std::numeric_limits<double>::infinity() : -std::log1p(-p) / rate;
}

double weibull_pdf(double x, double shape, double scale) {
    finite(x); positive(shape); positive(scale);
    if (x < 0) return 0;
    if (x == 0) {
        if (shape < 1) return std::numeric_limits<double>::infinity();
        return shape == 1 ? 1 / scale : 0;
    }
    const double z = x / scale;
    const double power = std::pow(z, shape);
    if (!std::isfinite(power)) return 0;
    return std::exp(std::log(shape) - std::log(scale) + (shape - 1) * std::log(z) - power);
}
double weibull_cdf(double x, double shape, double scale) {
    finite(x); positive(shape); positive(scale);
    return x < 0 ? 0 : -std::expm1(-std::pow(x / scale, shape));
}
double weibull_ppf(double p, double shape, double scale) {
    probability(p); positive(shape); positive(scale);
    return p == 1 ? std::numeric_limits<double>::infinity()
                  : scale * std::pow(-std::log1p(-p), 1 / shape);
}

double laplace_pdf(double x, double location, double scale) {
    finite(x); finite(location); positive(scale);
    return std::exp(-std::abs(x - location) / scale) / (2 * scale);
}
double laplace_cdf(double x, double location, double scale) {
    finite(x); finite(location); positive(scale);
    const double z = (x - location) / scale;
    return z < 0 ? 0.5 * std::exp(z) : -0.5 * std::expm1(-z) + 0.5;
}
double laplace_ppf(double p, double location, double scale) {
    probability(p); finite(location); positive(scale);
    if (p == 0) return -std::numeric_limits<double>::infinity();
    if (p == 1) return std::numeric_limits<double>::infinity();
    return p < 0.5 ? location + scale * std::log(2 * p)
                   : location - scale * std::log(2 * (1 - p));
}

double cauchy_pdf(double x, double location, double scale) {
    finite(x); finite(location); positive(scale);
    const double z = (x - location) / scale;
    return 1 / (pi * scale * (1 + z * z));
}
double cauchy_cdf(double x, double location, double scale) {
    finite(x); finite(location); positive(scale);
    return std::atan((x - location) / scale) / pi + 0.5;
}
double cauchy_ppf(double p, double location, double scale) {
    probability(p); finite(location); positive(scale);
    if (p == 0) return -std::numeric_limits<double>::infinity();
    if (p == 1) return std::numeric_limits<double>::infinity();
    return location + scale * std::tan(pi * (p - 0.5));
}

double lognormal_pdf(double x, double mu, double sigma) {
    finite(x); finite(mu); positive(sigma);
    if (x <= 0) return 0;
    const double z = (std::log(x) - mu) / sigma;
    return std::exp(normal_logpdf(z) - std::log(x) - std::log(sigma));
}
double lognormal_cdf(double x, double mu, double sigma) {
    finite(x); finite(mu); positive(sigma);
    return x <= 0 ? 0 : normal_cdf((std::log(x) - mu) / sigma);
}
double lognormal_ppf(double p, double mu, double sigma) {
    probability(p); finite(mu); positive(sigma);
    if (p == 0) return 0;
    if (p == 1) return std::numeric_limits<double>::infinity();
    return std::exp(mu + sigma * normal_ppf(p));
}
}  // namespace mathbr::distributions
