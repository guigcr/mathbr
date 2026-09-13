#pragma once

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

}  // namespace mathbr::distributions
