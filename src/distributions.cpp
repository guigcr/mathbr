#include "mathbr/distributions.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace mathbr::distributions {
namespace {
constexpr double inv_sqrt_2 = 0.70710678118654752440;
constexpr double log_sqrt_2pi = 0.91893853320467274178;

void validate_finite(double x) {
    if (!std::isfinite(x)) throw std::invalid_argument("x must be finite");
}

double uniform_width(double a, double b) {
    if (!std::isfinite(a) || !std::isfinite(b) || !(a < b) || !std::isfinite(b - a))
        throw std::invalid_argument("a and b must be finite with a < b and finite width");
    return b - a;
}

void validate_probability(double p) {
    if (!std::isfinite(p) || p < 0.0 || p > 1.0)
        throw std::invalid_argument("p must be finite and in [0, 1]");
}

double binomial_mass(int k, int n, double p) {
    if (k < 0 || k > n) return 0.0;
    if (p == 0.0) return k == 0 ? 1.0 : 0.0;
    if (p == 1.0) return k == n ? 1.0 : 0.0;
    const double log_mass = std::lgamma(n + 1.0) - std::lgamma(k + 1.0)
        - std::lgamma(n - k + 1.0) + k * std::log(p)
        + (n - k) * std::log1p(-p);
    return std::exp(log_mass);
}

void validate_df(double df) {
    if (!std::isfinite(df) || df <= 0.0)
        throw std::invalid_argument("df must be finite and positive");
}

double beta_fraction(double a, double b, double x) {
    constexpr double tiny = 1e-300;
    constexpr double tolerance = 4e-15;
    double c = 1.0;
    double d = 1.0 - (a + b) * x / (a + 1.0);
    if (std::abs(d) < tiny) d = tiny;
    d = 1.0 / d;
    double result = d;
    for (int m = 1; m <= 1000; ++m) {
        const double m2 = 2.0 * m;
        double step = m * (b - m) * x / ((a + m2 - 1.0) * (a + m2));
        d = 1.0 + step * d;
        if (std::abs(d) < tiny) d = tiny;
        c = 1.0 + step / c;
        if (std::abs(c) < tiny) c = tiny;
        d = 1.0 / d;
        result *= d * c;

        step = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0));
        d = 1.0 + step * d;
        if (std::abs(d) < tiny) d = tiny;
        c = 1.0 + step / c;
        if (std::abs(c) < tiny) c = tiny;
        d = 1.0 / d;
        const double delta = d * c;
        result *= delta;
        if (std::abs(delta - 1.0) < tolerance) return result;
    }
    throw std::runtime_error("incomplete beta fraction did not converge");
}

double beta_log_lower(double a, double b, double x) {
    if (x <= 0.0) return -std::numeric_limits<double>::infinity();
    if (x >= 1.0) return 0.0;
    const double log_front = std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b)
        + a * std::log(x) + b * std::log1p(-x);
    if (x < (a + 1.0) / (a + b + 2.0))
        return log_front + std::log(beta_fraction(a, b, x) / a);
    const double log_complement = log_front + std::log(beta_fraction(b, a, 1.0 - x) / b);
    return std::log1p(-std::exp(log_complement));
}

double student_t_log_lower_tail(double positive_x, double df) {
    const double ratio = positive_x / std::sqrt(df);
    if (ratio < 1.0) {
        const double y = ratio * ratio / (1.0 + ratio * ratio);
        const double log_center = beta_log_lower(0.5, df / 2.0, y);
        return -std::log(2.0) + std::log1p(-std::exp(log_center));
    }
    const double z = 1.0 / (1.0 + ratio * ratio);
    return -std::log(2.0) + beta_log_lower(df / 2.0, 0.5, z);
}

double gamma_regularized_lower(double a, double x) {
    if (x <= 0.0) return 0.0;
    const double log_factor = a * std::log(x) - x - std::lgamma(a);
    if (x < a + 1.0) {
        double term = 1.0 / a;
        double sum = term;
        for (int n = 1; n <= 1000; ++n) {
            term *= x / (a + n);
            sum += term;
            if (std::abs(term) <= std::abs(sum) * 4e-15)
                return std::fmin(1.0, std::exp(log_factor) * sum);
        }
    } else {
        constexpr double tiny = 1e-300;
        double b = x + 1.0 - a;
        double c = 1.0 / tiny;
        double d = 1.0 / (std::abs(b) < tiny ? tiny : b);
        double h = d;
        for (int i = 1; i <= 1000; ++i) {
            const double an = -i * (i - a);
            b += 2.0;
            d = an * d + b;
            if (std::abs(d) < tiny) d = tiny;
            c = b + an / c;
            if (std::abs(c) < tiny) c = tiny;
            d = 1.0 / d;
            const double delta = d * c;
            h *= delta;
            if (std::abs(delta - 1.0) < 4e-15)
                return std::fmax(0.0, 1.0 - std::exp(log_factor) * h);
        }
    }
    throw std::runtime_error("incomplete gamma calculation did not converge");
}
}  // namespace

double normal_logpdf(double x) {
    validate_finite(x);
    return -0.5 * x * x - log_sqrt_2pi;
}

double normal_pdf(double x) {
    return std::exp(normal_logpdf(x));
}

double normal_cdf(double x) {
    validate_finite(x);
    return 0.5 * std::erfc(-x * inv_sqrt_2);
}

double normal_logcdf(double x) {
    validate_finite(x);
    if (x > 0.0) return std::log1p(-normal_cdf(-x));
    if (x >= -10.0) return std::log(normal_cdf(x));

    // Mills-ratio asymptotic series, evaluated only where its terms decrease.
    const double t = -x;
    const double z = 1.0 / (t * t);
    const double correction = 1.0 + z * (-1.0 + z * (3.0 + z * (-15.0 + 105.0 * z)));
    return normal_logpdf(x) - std::log(t) + std::log(correction);
}

double normal_ppf(double p) {
    if (!std::isfinite(p) || p < 0.0 || p > 1.0)
        throw std::invalid_argument("p must be finite and in [0, 1]");
    if (p == 0.0) return -std::numeric_limits<double>::infinity();
    if (p == 1.0) return std::numeric_limits<double>::infinity();

    // Bisection stays stable even in tails where Newton steps can overshoot.
    double lo = -40.0;
    double hi = 40.0;
    for (int i = 0; i < 120; ++i) {
        const double mid = lo + 0.5 * (hi - lo);
        if (normal_cdf(mid) < p) lo = mid;
        else hi = mid;
    }
    return lo + 0.5 * (hi - lo);
}

double student_t_logpdf(double x, double df) {
    validate_finite(x);
    validate_df(df);
    return std::lgamma((df + 1.0) / 2.0) - std::lgamma(df / 2.0)
        - 0.5 * (std::log(df) + std::log(3.14159265358979323846))
        - (df + 1.0) / 2.0 * std::log1p((x / std::sqrt(df)) * (x / std::sqrt(df)));
}

double student_t_pdf(double x, double df) {
    return std::exp(student_t_logpdf(x, df));
}

double student_t_logcdf(double x, double df) {
    validate_finite(x);
    validate_df(df);
    const double log_tail = student_t_log_lower_tail(std::abs(x), df);
    return x <= 0.0 ? log_tail : std::log1p(-std::exp(log_tail));
}

double student_t_cdf(double x, double df) {
    return std::exp(student_t_logcdf(x, df));
}

double student_t_ppf(double p, double df) {
    validate_probability(p);
    validate_df(df);
    if (p == 0.0) return -std::numeric_limits<double>::infinity();
    if (p == 1.0) return std::numeric_limits<double>::infinity();
    if (p == 0.5) return 0.0;
    const double target = p < 0.5 ? p : 1.0 - p;
    double lo = 0.0;
    double hi = 1.0;
    while (std::exp(student_t_log_lower_tail(hi, df)) > target) {
        lo = hi;
        hi *= 2.0;
        if (!std::isfinite(hi)) return p < 0.5 ? -lo : lo;
    }
    for (int i = 0; i < 100; ++i) {
        const double mid = lo + 0.5 * (hi - lo);
        if (std::exp(student_t_log_lower_tail(mid, df)) > target) lo = mid;
        else hi = mid;
    }
    const double result = lo + 0.5 * (hi - lo);
    return p < 0.5 ? -result : result;
}

double chi_square_logpdf(double x, double df) {
    validate_finite(x);
    validate_df(df);
    if (x < 0.0) return -std::numeric_limits<double>::infinity();
    const double a = df / 2.0;
    if (x == 0.0) {
        if (a < 1.0) return std::numeric_limits<double>::infinity();
        if (a > 1.0) return -std::numeric_limits<double>::infinity();
        return -std::log(2.0);
    }
    return (a - 1.0) * std::log(x) - x / 2.0 - a * std::log(2.0) - std::lgamma(a);
}

double chi_square_pdf(double x, double df) { return std::exp(chi_square_logpdf(x, df)); }

double chi_square_cdf(double x, double df) {
    validate_finite(x);
    validate_df(df);
    return x <= 0.0 ? 0.0 : gamma_regularized_lower(df / 2.0, x / 2.0);
}

double chi_square_logcdf(double x, double df) { return std::log(chi_square_cdf(x, df)); }

double chi_square_ppf(double p, double df) {
    validate_probability(p);
    validate_df(df);
    if (p == 0.0) return 0.0;
    if (p == 1.0) return std::numeric_limits<double>::infinity();
    double lo = 0.0, hi = std::fmax(1.0, df);
    while (chi_square_cdf(hi, df) < p) {
        lo = hi;
        hi *= 2.0;
        if (!std::isfinite(hi)) return std::numeric_limits<double>::infinity();
    }
    for (int i = 0; i < 120; ++i) {
        const double mid = lo + 0.5 * (hi - lo);
        if (chi_square_cdf(mid, df) < p) lo = mid;
        else hi = mid;
    }
    return lo + 0.5 * (hi - lo);
}

double f_logpdf(double x, double df1, double df2) {
    validate_finite(x);
    validate_df(df1);
    validate_df(df2);
    if (x < 0.0) return -std::numeric_limits<double>::infinity();
    const double a = df1 / 2.0, b = df2 / 2.0;
    if (x == 0.0) {
        if (a < 1.0) return std::numeric_limits<double>::infinity();
        if (a > 1.0) return -std::numeric_limits<double>::infinity();
        return std::log(df1 / df2) - (std::lgamma(a) + std::lgamma(b)
                                     - std::lgamma(a + b));
    }
    const double ratio = df1 / df2;
    return a * std::log(ratio) + (a - 1.0) * std::log(x)
        - (a + b) * std::log1p(ratio * x)
        - (std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b));
}

double f_pdf(double x, double df1, double df2) {
    return std::exp(f_logpdf(x, df1, df2));
}

double f_cdf(double x, double df1, double df2) {
    validate_finite(x);
    validate_df(df1);
    validate_df(df2);
    if (x <= 0.0) return 0.0;
    const double scaled = (df1 / df2) * x;
    if (!std::isfinite(scaled)) return 1.0;
    const double z = scaled / (1.0 + scaled);
    return std::exp(beta_log_lower(df1 / 2.0, df2 / 2.0, z));
}

double f_logcdf(double x, double df1, double df2) {
    return std::log(f_cdf(x, df1, df2));
}

double f_ppf(double p, double df1, double df2) {
    validate_probability(p);
    validate_df(df1);
    validate_df(df2);
    if (p == 0.0) return 0.0;
    if (p == 1.0) return std::numeric_limits<double>::infinity();
    double lo = 0.0, hi = 1.0;
    while (f_cdf(hi, df1, df2) < p) {
        lo = hi;
        hi *= 2.0;
        if (!std::isfinite(hi)) return std::numeric_limits<double>::infinity();
    }
    for (int i = 0; i < 120; ++i) {
        const double mid = lo + 0.5 * (hi - lo);
        if (f_cdf(mid, df1, df2) < p) lo = mid;
        else hi = mid;
    }
    return lo + 0.5 * (hi - lo);
}

double uniform_logpdf(double x, double a, double b) {
    validate_finite(x);
    const double width = uniform_width(a, b);
    return x < a || x > b ? -std::numeric_limits<double>::infinity() : -std::log(width);
}

double uniform_pdf(double x, double a, double b) {
    return std::exp(uniform_logpdf(x, a, b));
}

double uniform_cdf(double x, double a, double b) {
    validate_finite(x);
    const double width = uniform_width(a, b);
    if (x <= a) return 0.0;
    if (x >= b) return 1.0;
    return (x - a) / width;
}

double uniform_logcdf(double x, double a, double b) {
    return std::log(uniform_cdf(x, a, b));
}

double uniform_ppf(double p, double a, double b) {
    if (!std::isfinite(p) || p < 0.0 || p > 1.0)
        throw std::invalid_argument("p must be finite and in [0, 1]");
    const double width = uniform_width(a, b);
    if (p == 0.0) return a;
    if (p == 1.0) return b;
    return a + p * width;
}

double bernoulli_pmf(int k, double p) {
    validate_probability(p);
    return binomial_mass(k, 1, p);
}

double bernoulli_cdf(int k, double p) {
    validate_probability(p);
    if (k < 0) return 0.0;
    if (k == 0) return 1.0 - p;
    return 1.0;
}

double binomial_pmf(int k, int n, double p) {
    validate_probability(p);
    if (n < 0) throw std::invalid_argument("n must be nonnegative");
    return binomial_mass(k, n, p);
}

double binomial_cdf(int k, int n, double p) {
    validate_probability(p);
    if (n < 0) throw std::invalid_argument("n must be nonnegative");
    if (k < 0) return 0.0;
    if (k >= n) return 1.0;
    if (p == 0.0) return 1.0;
    if (p == 1.0) return 0.0;

    // Sum the shorter tail to reduce work and cancellation near probability one.
    if (k <= n * p) {
        double total = 0.0;
        for (int j = 0; j <= k; ++j) total += binomial_mass(j, n, p);
        return std::fmin(1.0, total);
    }
    double tail = 0.0;
    for (int j = k + 1; j <= n; ++j) tail += binomial_mass(j, n, p);
    return std::fmax(0.0, 1.0 - tail);
}

double gamma_pdf(double x, double shape, double scale) {
    validate_finite(x); validate_df(shape); validate_df(scale);
    if (x > 0 && !std::isfinite(2 * x / scale)) return 0;
    return chi_square_pdf(2 * x / scale, 2 * shape) * 2 / scale;
}
double gamma_cdf(double x, double shape, double scale) {
    validate_finite(x); validate_df(shape); validate_df(scale);
    if (x > 0 && !std::isfinite(2 * x / scale)) return 1;
    return chi_square_cdf(2 * x / scale, 2 * shape);
}
double gamma_ppf(double p, double shape, double scale) {
    validate_probability(p); validate_df(shape); validate_df(scale);
    return chi_square_ppf(p, 2 * shape) * scale / 2;
}

double beta_pdf(double x, double alpha, double beta) {
    validate_finite(x); validate_df(alpha); validate_df(beta);
    if (x < 0 || x > 1) return 0;
    if (x == 0) {
        if (alpha < 1) return std::numeric_limits<double>::infinity();
        if (alpha > 1) return 0;
        return beta;
    }
    if (x == 1) {
        if (beta < 1) return std::numeric_limits<double>::infinity();
        if (beta > 1) return 0;
        return alpha;
    }
    const double log_density = (alpha - 1) * std::log(x)
        + (beta - 1) * std::log1p(-x)
        + std::lgamma(alpha + beta) - std::lgamma(alpha) - std::lgamma(beta);
    return std::exp(log_density);
}
double beta_cdf(double x, double alpha, double beta) {
    validate_finite(x); validate_df(alpha); validate_df(beta);
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    return std::exp(beta_log_lower(alpha, beta, x));
}
double beta_ppf(double p, double alpha, double beta) {
    validate_probability(p); validate_df(alpha); validate_df(beta);
    if (p == 0) return 0;
    if (p == 1) return 1;
    double lo = 0, hi = 1;
    for (int i = 0; i < 120; ++i) {
        const double mid = lo + (hi - lo) / 2;
        if (beta_cdf(mid, alpha, beta) < p) lo = mid;
        else hi = mid;
    }
    return lo + (hi - lo) / 2;
}

}  // namespace mathbr::distributions
