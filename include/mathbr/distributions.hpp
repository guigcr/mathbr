#pragma once

#include <vector>
#include <cstdint>

namespace mathbr::distributions {

double normal_pdf(double x);
double normal_logpdf(double x);
double normal_cdf(double x);
double normal_logcdf(double x);
double normal_ppf(double p);
double student_t_pdf(double x, double df);
double student_t_logpdf(double x, double df);
double student_t_cdf(double x, double df);
double student_t_logcdf(double x, double df);
double student_t_ppf(double p, double df);
double chi_square_pdf(double x, double df);
double chi_square_logpdf(double x, double df);
double chi_square_cdf(double x, double df);
double chi_square_logcdf(double x, double df);
double chi_square_ppf(double p, double df);
double f_pdf(double x, double df1, double df2);
double f_logpdf(double x, double df1, double df2);
double f_cdf(double x, double df1, double df2);
double f_logcdf(double x, double df1, double df2);
double f_ppf(double p, double df1, double df2);
double uniform_pdf(double x, double a, double b);
double uniform_logpdf(double x, double a, double b);
double uniform_cdf(double x, double a, double b);
double uniform_logcdf(double x, double a, double b);
double uniform_ppf(double p, double a, double b);
double bernoulli_pmf(int k, double p);
double bernoulli_cdf(int k, double p);
double binomial_pmf(int k, int n, double p);
double binomial_cdf(int k, int n, double p);
double binomial_ppf(double q, int n, double p);
double poisson_pmf(int k, double rate);
double poisson_cdf(int k, double rate);
double poisson_ppf(double q, double rate);
double negative_binomial_pmf(int k, int successes, double p);
double negative_binomial_cdf(int k, int successes, double p);
double negative_binomial_ppf(double q, int successes, double p);
double multinomial_pmf(const std::vector<int>& counts,
                       const std::vector<double>& probabilities);
std::vector<int> binomial_sample(std::size_t count, int n, double p, std::uint64_t seed);
std::vector<int> geometric_sample(std::size_t count, double p, std::uint64_t seed);
std::vector<int> poisson_sample(std::size_t count, double rate, std::uint64_t seed);
std::vector<int> negative_binomial_sample(std::size_t count, int successes,
                                          double p, std::uint64_t seed);
std::vector<int> multinomial_sample(int trials,
                                    const std::vector<double>& probabilities,
                                    std::uint64_t seed);
std::vector<int> bernoulli_sample(std::size_t count, double p, std::uint64_t seed);
std::vector<int> discrete_uniform_sample(std::size_t count, int a, int b,
                                         std::uint64_t seed);
std::vector<double> normal_sample(std::size_t count, std::uint64_t seed);
std::vector<double> student_t_sample(std::size_t count, double df, std::uint64_t seed);
std::vector<double> chi_square_sample(std::size_t count, double df, std::uint64_t seed);
std::vector<double> f_sample(std::size_t count, double df1, double df2, std::uint64_t seed);
std::vector<double> uniform_sample(std::size_t count, double a, double b, std::uint64_t seed);
std::vector<double> exponential_sample(std::size_t count, double rate, std::uint64_t seed);
std::vector<double> gamma_sample(std::size_t count, double shape, double scale,
                                 std::uint64_t seed);
std::vector<double> weibull_sample(std::size_t count, double shape, double scale,
                                   std::uint64_t seed);
std::vector<double> cauchy_sample(std::size_t count, double location, double scale,
                                  std::uint64_t seed);
std::vector<double> lognormal_sample(std::size_t count, double mu, double sigma,
                                    std::uint64_t seed);
std::vector<double> laplace_sample(std::size_t count, double location, double scale,
                                  std::uint64_t seed);
std::vector<double> beta_sample(std::size_t count, double alpha, double beta,
                               std::uint64_t seed);
double geometric_pmf(int k, double p);
double geometric_cdf(int k, double p);
double geometric_ppf(double q, double p);
double discrete_uniform_pmf(int k, int a, int b);
double discrete_uniform_cdf(int k, int a, int b);
double discrete_uniform_ppf(double q, int a, int b);
double exponential_pdf(double x, double rate);
double exponential_cdf(double x, double rate);
double exponential_ppf(double p, double rate);
double weibull_pdf(double x, double shape, double scale);
double weibull_cdf(double x, double shape, double scale);
double weibull_ppf(double p, double shape, double scale);
double laplace_pdf(double x, double location, double scale);
double laplace_cdf(double x, double location, double scale);
double laplace_ppf(double p, double location, double scale);
double cauchy_pdf(double x, double location, double scale);
double cauchy_cdf(double x, double location, double scale);
double cauchy_ppf(double p, double location, double scale);
double lognormal_pdf(double x, double mu, double sigma);
double lognormal_cdf(double x, double mu, double sigma);
double lognormal_ppf(double p, double mu, double sigma);
double gamma_pdf(double x, double shape, double scale);
double gamma_cdf(double x, double shape, double scale);
double gamma_ppf(double p, double shape, double scale);
double beta_pdf(double x, double alpha, double beta);
double beta_cdf(double x, double alpha, double beta);
double beta_ppf(double p, double alpha, double beta);
double pareto_pdf(double x, double shape, double scale);
double pareto_cdf(double x, double shape, double scale);
double pareto_ppf(double p, double shape, double scale);
std::vector<double> pareto_sample(std::size_t count, double shape, double scale,
                                  std::uint64_t seed);
double dirichlet_logpdf(const std::vector<double>& x, const std::vector<double>& alpha);
double dirichlet_pdf(const std::vector<double>& x, const std::vector<double>& alpha);
std::vector<std::vector<double>> dirichlet_sample(std::size_t count,
                                                  const std::vector<double>& alpha,
                                                  std::uint64_t seed);
std::vector<double> normal_fit(const std::vector<double>& observations);
double exponential_fit(const std::vector<double>& observations);
double poisson_fit(const std::vector<int>& observations);

}  // namespace mathbr::distributions
