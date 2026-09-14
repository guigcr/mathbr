#include "mathbr/distributions.hpp"

#include <cmath>
#include <limits>
#include <cstdint>
#include <stdexcept>
#include <random>
#include <algorithm>

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
template <typename Distribution>
std::vector<double> draw_samples(std::size_t count, std::uint64_t seed,
                                 Distribution distribution) {
    std::mt19937_64 generator(seed);
    std::vector<double> result(count);
    for (double& value : result) {
        value = distribution(generator);
        if (!std::isfinite(value)) throw std::overflow_error("sample is not finite");
    }
    return result;
}
}  // namespace

std::vector<double> normal_sample(std::size_t count, std::uint64_t seed) {
    return draw_samples(count, seed, std::normal_distribution<double>(0.0, 1.0));
}

std::vector<double> student_t_sample(std::size_t count, double df, std::uint64_t seed) {
    positive(df);
    return draw_samples(count, seed, std::student_t_distribution<double>(df));
}

std::vector<double> chi_square_sample(std::size_t count, double df, std::uint64_t seed) {
    positive(df);
    return draw_samples(count, seed, std::chi_squared_distribution<double>(df));
}

std::vector<double> f_sample(std::size_t count, double df1, double df2,
                             std::uint64_t seed) {
    positive(df1); positive(df2);
    return draw_samples(count, seed, std::fisher_f_distribution<double>(df1, df2));
}

std::vector<double> uniform_sample(std::size_t count, double a, double b,
                                   std::uint64_t seed) {
    finite(a); finite(b);
    if (!(a < b) || !std::isfinite(b - a))
        throw std::invalid_argument("a and b must have positive finite width");
    return draw_samples(count, seed, std::uniform_real_distribution<double>(a, b));
}

std::vector<double> exponential_sample(std::size_t count, double rate,
                                       std::uint64_t seed) {
    positive(rate);
    return draw_samples(count, seed, std::exponential_distribution<double>(rate));
}

std::vector<double> gamma_sample(std::size_t count, double shape, double scale,
                                 std::uint64_t seed) {
    positive(shape); positive(scale);
    return draw_samples(count, seed, std::gamma_distribution<double>(shape, scale));
}

std::vector<double> weibull_sample(std::size_t count, double shape, double scale,
                                   std::uint64_t seed) {
    positive(shape); positive(scale);
    return draw_samples(count, seed, std::weibull_distribution<double>(shape, scale));
}

std::vector<double> cauchy_sample(std::size_t count, double location, double scale,
                                  std::uint64_t seed) {
    finite(location); positive(scale);
    return draw_samples(count, seed, std::cauchy_distribution<double>(location, scale));
}

std::vector<double> lognormal_sample(std::size_t count, double mu, double sigma,
                                    std::uint64_t seed) {
    finite(mu); positive(sigma);
    return draw_samples(count, seed, std::lognormal_distribution<double>(mu, sigma));
}

std::vector<double> laplace_sample(std::size_t count, double location, double scale,
                                  std::uint64_t seed) {
    finite(location); positive(scale);
    std::mt19937_64 generator(seed);
    std::exponential_distribution<double> magnitude(1.0);
    std::bernoulli_distribution sign(0.5);
    std::vector<double> result(count);
    for (double& value : result) {
        const double step = scale * magnitude(generator);
        value = location + (sign(generator) ? step : -step);
        if (!std::isfinite(value)) throw std::overflow_error("sample is not finite");
    }
    return result;
}

std::vector<double> beta_sample(std::size_t count, double alpha, double beta,
                                std::uint64_t seed) {
    positive(alpha); positive(beta);
    std::mt19937_64 generator(seed);
    std::gamma_distribution<double> first(alpha, 1.0), second(beta, 1.0);
    std::vector<double> result(count);
    for (double& value : result) {
        const double x = first(generator), y = second(generator);
        const double maximum = std::max(x, y);
        if (!(maximum > 0.0) || !std::isfinite(maximum))
            throw std::overflow_error("beta gamma variates are not finite and positive");
        value = (x / maximum) / (x / maximum + y / maximum);
    }
    return result;
}

double pareto_pdf(double x, double shape, double scale) {
    finite(x); positive(shape); positive(scale);
    if (x < scale) return 0.0;
    return std::exp(std::log(shape) - std::log(x)
                    + shape * (std::log(scale) - std::log(x)));
}

double pareto_cdf(double x, double shape, double scale) {
    finite(x); positive(shape); positive(scale);
    if (x < scale) return 0.0;
    return -std::expm1(shape * (std::log(scale) - std::log(x)));
}

double pareto_ppf(double p, double shape, double scale) {
    probability(p); positive(shape); positive(scale);
    if (p == 0.0) return scale;
    if (p == 1.0) return std::numeric_limits<double>::infinity();
    return scale * std::exp(-std::log1p(-p) / shape);
}

std::vector<double> pareto_sample(std::size_t count, double shape, double scale,
                                  std::uint64_t seed) {
    positive(shape); positive(scale);
    std::mt19937_64 generator(seed);
    std::exponential_distribution<double> log_ratio(shape);
    std::vector<double> result(count);
    for (double& value : result) {
        value = scale * std::exp(log_ratio(generator));
        if (!std::isfinite(value)) throw std::overflow_error("sample is not finite");
    }
    return result;
}

double dirichlet_logpdf(const std::vector<double>& x, const std::vector<double>& alpha) {
    if (x.size() < 2 || x.size() != alpha.size())
        throw std::invalid_argument("Dirichlet vectors must have the same length of at least two");
    long double x_sum = 0.0, alpha_sum = 0.0;
    double result = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (!std::isfinite(x[i]) || x[i] <= 0.0 || !std::isfinite(alpha[i]) || alpha[i] <= 0.0)
            throw std::invalid_argument("Dirichlet x and alpha must be finite and positive");
        x_sum += x[i];
        alpha_sum += alpha[i];
        result += (alpha[i] - 1.0) * std::log(x[i]) - std::lgamma(alpha[i]);
    }
    if (std::abs(x_sum - 1.0L) > 1e-12L || alpha_sum > std::numeric_limits<double>::max())
        throw std::invalid_argument("x must sum to one and alpha sum must be finite");
    return result + std::lgamma(static_cast<double>(alpha_sum));
}

double dirichlet_pdf(const std::vector<double>& x, const std::vector<double>& alpha) {
    return std::exp(dirichlet_logpdf(x, alpha));
}

std::vector<std::vector<double>> dirichlet_sample(std::size_t count,
                                                  const std::vector<double>& alpha,
                                                  std::uint64_t seed) {
    if (alpha.size() < 2) throw std::invalid_argument("alpha needs at least two values");
    std::vector<std::gamma_distribution<double>> distributions;
    distributions.reserve(alpha.size());
    long double alpha_sum = 0.0;
    for (double value : alpha) {
        positive(value);
        alpha_sum += value;
        distributions.emplace_back(value, 1.0);
    }
    if (alpha_sum > std::numeric_limits<double>::max())
        throw std::invalid_argument("alpha sum must be finite");
    std::mt19937_64 generator(seed);
    std::vector<std::vector<double>> result(count, std::vector<double>(alpha.size()));
    for (auto& row : result) {
        double maximum = 0.0;
        for (std::size_t i = 0; i < alpha.size(); ++i) {
            row[i] = distributions[i](generator);
            maximum = std::max(maximum, row[i]);
        }
        if (!(maximum > 0.0) || !std::isfinite(maximum))
            throw std::overflow_error("Dirichlet gamma variates are not finite and positive");
        double total = 0.0;
        for (double& value : row) {
            value /= maximum;
            total += value;
        }
        for (double& value : row) value /= total;
    }
    return result;
}

std::vector<double> normal_fit(const std::vector<double>& observations) {
    if (observations.empty()) throw std::invalid_argument("observations must be nonempty");
    const double reference = observations.front();
    double offset = 0.0;
    for (double value : observations) {
        finite(value);
        offset += value / observations.size() - reference / observations.size();
    }
    const double mean = reference + offset;
    if (!std::isfinite(mean)) throw std::invalid_argument("fitted mean is not finite");
    double maximum = 0.0;
    for (double value : observations)
        maximum = std::max(maximum, std::abs(value - mean));
    if (!(maximum > 0.0) || !std::isfinite(maximum))
        throw std::invalid_argument("normal fit requires varying finite data");
    double scaled_sum = 0.0;
    for (double value : observations) {
        const double scaled = (value - mean) / maximum;
        scaled_sum += scaled * scaled;
    }
    const double sigma = maximum * std::sqrt(scaled_sum / observations.size());
    if (!(sigma > 0.0) || !std::isfinite(sigma))
        throw std::invalid_argument("fitted standard deviation is not finite and positive");
    return {mean, sigma};
}

double exponential_fit(const std::vector<double>& observations) {
    if (observations.empty()) throw std::invalid_argument("observations must be nonempty");
    double mean = 0.0;
    for (double value : observations) {
        finite(value);
        if (value < 0.0) throw std::invalid_argument("exponential observations must be nonnegative");
        mean += value / observations.size();
    }
    if (!(mean > 0.0) || !std::isfinite(mean))
        throw std::invalid_argument("exponential fit needs a positive finite mean");
    const double rate = 1.0 / mean;
    if (!std::isfinite(rate)) throw std::invalid_argument("fitted rate is not finite");
    return rate;
}

double geometric_pmf(int k, double p) {
    positive(p);
    if (p > 1) throw std::invalid_argument("p must be in (0, 1]");
    if (k < 1) return 0.0;
    if (p == 1) return k == 1 ? 1.0 : 0.0;
    return p * std::exp((static_cast<double>(k) - 1) * std::log1p(-p));
}

double geometric_cdf(int k, double p) {
    positive(p);
    if (p > 1) throw std::invalid_argument("p must be in (0, 1]");
    if (k < 1) return 0.0;
    if (p == 1) return 1.0;
    if (k == 1) return p;
    return -std::expm1(static_cast<double>(k) * std::log1p(-p));
}

double geometric_ppf(double q, double p) {
    probability(q); positive(p);
    if (p > 1) throw std::invalid_argument("p must be in (0, 1]");
    if (q == 0 || p == 1) return 1.0;
    if (q == 1) return std::numeric_limits<double>::infinity();
    double k = std::max(1.0, std::ceil(std::log1p(-q) / std::log1p(-p)));
    if (k <= std::numeric_limits<int>::max()) {
        while (k > 1 && geometric_cdf(static_cast<int>(k - 1), p) >= q) --k;
        while (k < std::numeric_limits<int>::max() && geometric_cdf(static_cast<int>(k), p) < q) ++k;
    }
    return k;
}

double discrete_uniform_pmf(int k, int a, int b) {
    if (a > b) throw std::invalid_argument("a must not exceed b");
    const auto count = static_cast<std::int64_t>(b) - a + 1;
    return k < a || k > b ? 0.0 : 1.0 / static_cast<double>(count);
}

double discrete_uniform_cdf(int k, int a, int b) {
    if (a > b) throw std::invalid_argument("a must not exceed b");
    if (k < a) return 0.0;
    if (k >= b) return 1.0;
    const auto count = static_cast<std::int64_t>(b) - a + 1;
    const auto through = static_cast<std::int64_t>(k) - a + 1;
    return static_cast<double>(through) / static_cast<double>(count);
}

double discrete_uniform_ppf(double q, int a, int b) {
    probability(q);
    if (a > b) throw std::invalid_argument("a must not exceed b");
    if (q == 0) return a;
    if (q == 1) return b;
    const auto count = static_cast<std::int64_t>(b) - a + 1;
    auto offset = static_cast<std::int64_t>(std::ceil(q * static_cast<double>(count))) - 1;
    if (offset < 0) offset = 0;
    if (offset >= count) offset = count - 1;
    return static_cast<double>(static_cast<std::int64_t>(a) + offset);
}

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
