#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include "mathbr/ols.hpp"
#include "mathbr/regularized.hpp"
#include "mathbr/wls.hpp"
#include "mathbr/econometrics.hpp"
#include "mathbr/time_series.hpp"
#include "mathbr/distributions.hpp"
#include "mathbr/statistics.hpp"
#include "mathbr/hypothesis.hpp"
#include "mathbr/evaluation.hpp"
#include "mathbr/diagnostics.hpp"
#include "mathbr/nonparametric.hpp"
#include "mathbr/time_series_diagnostics.hpp"
#include "mathbr/survival.hpp"
#include "mathbr/bayesian.hpp"
#include "mathbr/multivariate.hpp"

namespace py = pybind11;

namespace Validation {
    void paired_size(size_t actual, size_t expected) {
        if (actual != expected || expected == 0) {
            throw std::invalid_argument("inputs must have the same nonzero length");
        }
    }

    void training_data(const std::vector<std::vector<double>>& X, size_t y_size,
                       size_t n_features, double lr, int epochs) {
        paired_size(X.size(), y_size);
        if (n_features == 0 || !std::isfinite(lr) || lr <= 0.0 || epochs <= 0) {
            throw std::invalid_argument("n_features, lr and epochs must be positive (lr finite)");
        }
        for (const auto& row : X) {
            if (row.size() != n_features) {
                throw std::invalid_argument("each X row must match n_features");
            }
        }
    }
}

// activation functions
namespace Activations {

    double sigmoid(double z) {
        if (z >= 0.0) return 1.0 / (1.0 + std::exp(-z));
        const double e = std::exp(z);
        return e / (1.0 + e);
    }

    double sigmoid_derivative(double z) {
        double s = sigmoid(z);
        return s * (1.0 - s);
    }

    double tanh_activation(double z) {
        return std::tanh(z);
    }

    double tanh_derivative(double z) {
        double t = std::tanh(z);
        return 1.0 - t * t;
    }

    double relu(double z) {
        return std::max(0.0, z);
    }

    double relu_derivative(double z) {
        return z > 0.0 ? 1.0 : 0.0;
    }

    double leaky_relu(double z, double alpha = 0.01) {
        return z > 0 ? z : alpha * z;
    }

    double leaky_relu_derivative(double z, double alpha = 0.01) {
        return z > 0.0 ? 1.0 : alpha;
    }

    double gelu(double z) {
        // tanh approximation, same one used in BERT/GPT
        const double c = std::sqrt(2.0 / 3.14159265358979323846);
        return 0.5 * z * (1.0 + std::tanh(c * (z + 0.044715 * std::pow(z, 3))));
    }

    double gelu_derivative(double z) {
        const double c = std::sqrt(2.0 / 3.14159265358979323846);
        double inner = c * (z + 0.044715 * std::pow(z, 3));
        double t = std::tanh(inner);
        double dt_dz = c * (1.0 + 3.0 * 0.044715 * z * z);
        double sech2 = 1.0 - t * t;
        return 0.5 * (1.0 + t) + 0.5 * z * sech2 * dt_dz;
    }

    double swish(double z) {
        return z * sigmoid(z);
    }

    double swish_derivative(double z) {
        double s = sigmoid(z);
        return s + z * s * (1.0 - s);
    }

    std::vector<double> softmax(const std::vector<double>& z) {
        if (z.empty()) throw std::invalid_argument("softmax requires a nonempty vector");
        std::vector<double> out(z.size());
        double max_val = *std::max_element(z.begin(), z.end());
        double sum = 0.0;
        for (size_t i = 0; i < z.size(); i++) {
            out[i] = std::exp(z[i] - max_val); // subtract max for numerical stability
            sum += out[i];
        }
        for (size_t i = 0; i < z.size(); i++) {
            out[i] /= sum;
        }
        return out;
    }

}


// loss functions and metrics
namespace Losses {

    double mse(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
        Validation::paired_size(y_true.size(), y_pred.size());
        double total = 0.0;
        int n = y_true.size();
        for (int i = 0; i < n; i++) {
            total += std::pow(y_true[i] - y_pred[i], 2);
        }
        return total / n;
    }

    std::vector<double> mse_derivative(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
        Validation::paired_size(y_true.size(), y_pred.size());
        int n = y_true.size();
        std::vector<double> grad(n);
        for (int i = 0; i < n; i++) {
            grad[i] = 2.0 * (y_pred[i] - y_true[i]) / n;
        }
        return grad;
    }

    double mae(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
        Validation::paired_size(y_true.size(), y_pred.size());
        double total = 0.0;
        int n = y_true.size();
        for (int i = 0; i < n; i++) {
            total += std::fabs(y_true[i] - y_pred[i]);
        }
        return total / n;
    }

    double rmse(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
        return std::sqrt(mse(y_true, y_pred));
    }

    double logloss(const std::vector<int>& y_true, const std::vector<double>& y_pred) {
        Validation::paired_size(y_true.size(), y_pred.size());
        double loss = 0.0;
        int n = y_true.size();
        for (int i = 0; i < n; i++) {
            if (y_true[i] != 0 && y_true[i] != 1) {
                throw std::invalid_argument("logloss labels must be 0 or 1");
            }
            if (!std::isfinite(y_pred[i]) || y_pred[i] < 0.0 || y_pred[i] > 1.0) {
                throw std::invalid_argument("logloss probabilities must be finite and in [0, 1]");
            }
            double p = std::clamp(y_pred[i], 1e-15, 1.0 - 1e-15);
            loss += y_true[i] * std::log(p) + (1 - y_true[i]) * std::log(1.0 - p);
        }
        return -loss / n;
    }

}


// linear regression
class LinearRegression {
private:
    std::vector<double> w;
    double b;
    bool is_trained = false;

public:
    LinearRegression(int n_features) : w(n_features > 0 ? n_features : 0, 0.0), b(0.0) {
        if (n_features <= 0) throw std::invalid_argument("n_features must be positive");
    }

    double predict(const std::vector<double>& x) const {
        if ((int)x.size() != (int)w.size()) {
            throw std::invalid_argument("x size doesn't match number of weights");
        }
        double y = b;
        for (size_t i = 0; i < x.size(); i++) {
            y += x[i] * w[i];
        }
        return y;
    }

    std::vector<double> predict_batch(const std::vector<std::vector<double>>& X) const {
        std::vector<double> out(X.size());
        for (size_t i = 0; i < X.size(); i++) {
            out[i] = predict(X[i]);
        }
        return out;
    }

    // trains with gradient descent using MSE
    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y,
              double lr = 0.01, int epochs = 1000) {
        Validation::training_data(X, y.size(), w.size(), lr, epochs);

        int n = X.size();
        int n_features = w.size();
        const double scale = 2.0 / n;
        std::vector<double> grad_w(n_features);

        for (int epoch = 0; epoch < epochs; epoch++) {
            std::fill(grad_w.begin(), grad_w.end(), 0.0);
            double grad_b = 0.0;

            for (int i = 0; i < n; i++) {
                double prediction = b;
                for (int j = 0; j < n_features; j++) prediction += X[i][j] * w[j];
                const double error = (prediction - y[i]) * scale;
                for (int j = 0; j < n_features; j++) {
                    grad_w[j] += error * X[i][j];
                }
                grad_b += error;
            }

            for (int j = 0; j < n_features; j++) {
                w[j] -= lr * grad_w[j];
            }
            b -= lr * grad_b;
        }
        is_trained = true;
    }

    std::vector<double> get_weights() const { return w; }
    double get_bias() const { return b; }
    bool trained() const { return is_trained; }
};


// logistic regression
class LogisticRegression {
private:
    std::vector<double> w;
    double b;
    bool is_trained = false;

public:
    LogisticRegression(int n_features) : w(n_features > 0 ? n_features : 0, 0.0), b(0.0) {
        if (n_features <= 0) throw std::invalid_argument("n_features must be positive");
    }

    double predict_proba(const std::vector<double>& x) const {
        if ((int)x.size() != (int)w.size()) {
            throw std::invalid_argument("x size doesn't match number of weights");
        }
        double z = b;
        for (size_t i = 0; i < x.size(); i++) {
            z += x[i] * w[i];
        }
        return Activations::sigmoid(z);
    }

    int predict(const std::vector<double>& x, double threshold = 0.5) const {
        if (!std::isfinite(threshold) || threshold < 0.0 || threshold > 1.0) {
            throw std::invalid_argument("threshold must be in [0, 1]");
        }
        return predict_proba(x) >= threshold ? 1 : 0;
    }

    std::vector<double> predict_proba_batch(const std::vector<std::vector<double>>& X) const {
        std::vector<double> out(X.size());
        for (size_t i = 0; i < X.size(); i++) {
            out[i] = predict_proba(X[i]);
        }
        return out;
    }

    // trains with gradient descent using log loss
    // note: sigmoid + logloss gradient simplifies to (p - y)
    void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y,
              double lr = 0.01, int epochs = 1000) {
        Validation::training_data(X, y.size(), w.size(), lr, epochs);
        for (int label : y) {
            if (label != 0 && label != 1) throw std::invalid_argument("labels must be 0 or 1");
        }

        int n = X.size();
        int n_features = w.size();
        const double scale = 1.0 / n;
        std::vector<double> grad_w(n_features);

        for (int epoch = 0; epoch < epochs; epoch++) {
            std::fill(grad_w.begin(), grad_w.end(), 0.0);
            double grad_b = 0.0;

            for (int i = 0; i < n; i++) {
                double score = b;
                for (int j = 0; j < n_features; j++) score += X[i][j] * w[j];
                const double error = Activations::sigmoid(score) - y[i];
                for (int j = 0; j < n_features; j++) {
                    grad_w[j] += error * X[i][j];
                }
                grad_b += error;
            }

            for (int j = 0; j < n_features; j++) {
                w[j] -= lr * grad_w[j] * scale;
            }
            b -= lr * grad_b * scale;
        }
        is_trained = true;
    }

    std::vector<double> get_weights() const { return w; }
    double get_bias() const { return b; }
    bool trained() const { return is_trained; }
};


PYBIND11_MODULE(mathbr, m) {
    m.doc() = "Educational machine-learning math library";
    m.attr("version") = "0.6.0";

    py::module_ distributions = m.def_submodule("distributions", "Probability distributions");
    py::class_<mathbr::multivariate::MultivariateNormal>(distributions, "MultivariateNormal")
        .def(py::init<const std::vector<double>&, const std::vector<std::vector<double>>&>(),
             py::arg("mean"), py::arg("covariance"))
        .def("logpdf", &mathbr::multivariate::MultivariateNormal::logpdf, py::arg("x"))
        .def("pdf", &mathbr::multivariate::MultivariateNormal::pdf, py::arg("x"))
        .def("mahalanobis_distance", &mathbr::multivariate::MultivariateNormal::mahalanobis_distance,
             py::arg("x"))
        .def("logpdf_batch", &mathbr::multivariate::MultivariateNormal::logpdf_batch,
             py::arg("data"))
        .def("pdf_batch", &mathbr::multivariate::MultivariateNormal::pdf_batch,
             py::arg("data"))
        .def("sample", &mathbr::multivariate::MultivariateNormal::sample,
             py::arg("count"), py::arg("seed"));
    distributions.def("normal_pdf", &mathbr::distributions::normal_pdf, py::arg("x"));
    distributions.def("normal_logpdf", &mathbr::distributions::normal_logpdf, py::arg("x"));
    distributions.def("normal_cdf", &mathbr::distributions::normal_cdf, py::arg("x"));
    distributions.def("normal_logcdf", &mathbr::distributions::normal_logcdf, py::arg("x"));
    distributions.def("normal_ppf", &mathbr::distributions::normal_ppf, py::arg("p"));
    distributions.def("student_t_pdf", &mathbr::distributions::student_t_pdf,
                      py::arg("x"), py::arg("df"));
    distributions.def("student_t_logpdf", &mathbr::distributions::student_t_logpdf,
                      py::arg("x"), py::arg("df"));
    distributions.def("student_t_cdf", &mathbr::distributions::student_t_cdf,
                      py::arg("x"), py::arg("df"));
    distributions.def("student_t_logcdf", &mathbr::distributions::student_t_logcdf,
                      py::arg("x"), py::arg("df"));
    distributions.def("student_t_ppf", &mathbr::distributions::student_t_ppf,
                      py::arg("p"), py::arg("df"));
    distributions.def("chi_square_pdf", &mathbr::distributions::chi_square_pdf,
                      py::arg("x"), py::arg("df"));
    distributions.def("chi_square_logpdf", &mathbr::distributions::chi_square_logpdf,
                      py::arg("x"), py::arg("df"));
    distributions.def("chi_square_cdf", &mathbr::distributions::chi_square_cdf,
                      py::arg("x"), py::arg("df"));
    distributions.def("chi_square_logcdf", &mathbr::distributions::chi_square_logcdf,
                      py::arg("x"), py::arg("df"));
    distributions.def("chi_square_ppf", &mathbr::distributions::chi_square_ppf,
                      py::arg("p"), py::arg("df"));
    distributions.def("f_pdf", &mathbr::distributions::f_pdf,
                      py::arg("x"), py::arg("df1"), py::arg("df2"));
    distributions.def("f_logpdf", &mathbr::distributions::f_logpdf,
                      py::arg("x"), py::arg("df1"), py::arg("df2"));
    distributions.def("f_cdf", &mathbr::distributions::f_cdf,
                      py::arg("x"), py::arg("df1"), py::arg("df2"));
    distributions.def("f_logcdf", &mathbr::distributions::f_logcdf,
                      py::arg("x"), py::arg("df1"), py::arg("df2"));
    distributions.def("f_ppf", &mathbr::distributions::f_ppf,
                      py::arg("p"), py::arg("df1"), py::arg("df2"));
    distributions.def("uniform_pdf", &mathbr::distributions::uniform_pdf,
                      py::arg("x"), py::arg("a"), py::arg("b"));
    distributions.def("uniform_logpdf", &mathbr::distributions::uniform_logpdf,
                      py::arg("x"), py::arg("a"), py::arg("b"));
    distributions.def("uniform_cdf", &mathbr::distributions::uniform_cdf,
                      py::arg("x"), py::arg("a"), py::arg("b"));
    distributions.def("uniform_logcdf", &mathbr::distributions::uniform_logcdf,
                      py::arg("x"), py::arg("a"), py::arg("b"));
    distributions.def("uniform_ppf", &mathbr::distributions::uniform_ppf,
                      py::arg("p"), py::arg("a"), py::arg("b"));
    distributions.def("bernoulli_pmf", &mathbr::distributions::bernoulli_pmf,
                      py::arg("k"), py::arg("p"));
    distributions.def("bernoulli_cdf", &mathbr::distributions::bernoulli_cdf,
                      py::arg("k"), py::arg("p"));
    distributions.def("binomial_pmf", &mathbr::distributions::binomial_pmf,
                      py::arg("k"), py::arg("n"), py::arg("p"));
    distributions.def("binomial_cdf", &mathbr::distributions::binomial_cdf,
                      py::arg("k"), py::arg("n"), py::arg("p"));
    distributions.def("binomial_ppf", &mathbr::distributions::binomial_ppf,
                      py::arg("q"), py::arg("n"), py::arg("p"));
    distributions.def("poisson_pmf", &mathbr::distributions::poisson_pmf,
                      py::arg("k"), py::arg("rate"));
    distributions.def("poisson_cdf", &mathbr::distributions::poisson_cdf,
                      py::arg("k"), py::arg("rate"));
    distributions.def("poisson_ppf", &mathbr::distributions::poisson_ppf,
                      py::arg("q"), py::arg("rate"));
    distributions.def("negative_binomial_pmf", &mathbr::distributions::negative_binomial_pmf,
                      py::arg("k"), py::arg("successes"), py::arg("p"));
    distributions.def("negative_binomial_cdf", &mathbr::distributions::negative_binomial_cdf,
                      py::arg("k"), py::arg("successes"), py::arg("p"));
    distributions.def("negative_binomial_ppf", &mathbr::distributions::negative_binomial_ppf,
                      py::arg("q"), py::arg("successes"), py::arg("p"));
    distributions.def("multinomial_pmf", &mathbr::distributions::multinomial_pmf,
                      py::arg("counts"), py::arg("probabilities"));
    distributions.def("binomial_sample", &mathbr::distributions::binomial_sample,
                      py::arg("count"), py::arg("n"), py::arg("p"), py::arg("seed"));
    distributions.def("geometric_sample", &mathbr::distributions::geometric_sample,
                      py::arg("count"), py::arg("p"), py::arg("seed"));
    distributions.def("poisson_sample", &mathbr::distributions::poisson_sample,
                      py::arg("count"), py::arg("rate"), py::arg("seed"));
    distributions.def("negative_binomial_sample", &mathbr::distributions::negative_binomial_sample,
                      py::arg("count"), py::arg("successes"), py::arg("p"), py::arg("seed"));
    distributions.def("multinomial_sample", &mathbr::distributions::multinomial_sample,
                      py::arg("trials"), py::arg("probabilities"), py::arg("seed"));
    distributions.def("bernoulli_sample", &mathbr::distributions::bernoulli_sample,
                      py::arg("count"), py::arg("p"), py::arg("seed"));
    distributions.def("discrete_uniform_sample", &mathbr::distributions::discrete_uniform_sample,
                      py::arg("count"), py::arg("a"), py::arg("b"), py::arg("seed"));
    distributions.def("normal_sample", &mathbr::distributions::normal_sample,
                      py::arg("count"), py::arg("seed"));
    distributions.def("student_t_sample", &mathbr::distributions::student_t_sample,
                      py::arg("count"), py::arg("df"), py::arg("seed"));
    distributions.def("chi_square_sample", &mathbr::distributions::chi_square_sample,
                      py::arg("count"), py::arg("df"), py::arg("seed"));
    distributions.def("f_sample", &mathbr::distributions::f_sample,
                      py::arg("count"), py::arg("df1"), py::arg("df2"), py::arg("seed"));
    distributions.def("uniform_sample", &mathbr::distributions::uniform_sample,
                      py::arg("count"), py::arg("a"), py::arg("b"), py::arg("seed"));
    distributions.def("exponential_sample", &mathbr::distributions::exponential_sample,
                      py::arg("count"), py::arg("rate"), py::arg("seed"));
    distributions.def("gamma_sample", &mathbr::distributions::gamma_sample,
                      py::arg("count"), py::arg("shape"), py::arg("scale"), py::arg("seed"));
    distributions.def("weibull_sample", &mathbr::distributions::weibull_sample,
                      py::arg("count"), py::arg("shape"), py::arg("scale"), py::arg("seed"));
    distributions.def("cauchy_sample", &mathbr::distributions::cauchy_sample,
                      py::arg("count"), py::arg("location"), py::arg("scale"), py::arg("seed"));
    distributions.def("lognormal_sample", &mathbr::distributions::lognormal_sample,
                      py::arg("count"), py::arg("mu"), py::arg("sigma"), py::arg("seed"));
    distributions.def("laplace_sample", &mathbr::distributions::laplace_sample,
                      py::arg("count"), py::arg("location"), py::arg("scale"), py::arg("seed"));
    distributions.def("beta_sample", &mathbr::distributions::beta_sample,
                      py::arg("count"), py::arg("alpha"), py::arg("beta"), py::arg("seed"));
    distributions.def("geometric_pmf", &mathbr::distributions::geometric_pmf,
                      py::arg("k"), py::arg("p"));
    distributions.def("geometric_cdf", &mathbr::distributions::geometric_cdf,
                      py::arg("k"), py::arg("p"));
    distributions.def("geometric_ppf", &mathbr::distributions::geometric_ppf,
                      py::arg("q"), py::arg("p"));
    distributions.def("discrete_uniform_pmf", &mathbr::distributions::discrete_uniform_pmf,
                      py::arg("k"), py::arg("a"), py::arg("b"));
    distributions.def("discrete_uniform_cdf", &mathbr::distributions::discrete_uniform_cdf,
                      py::arg("k"), py::arg("a"), py::arg("b"));
    distributions.def("discrete_uniform_ppf", &mathbr::distributions::discrete_uniform_ppf,
                      py::arg("q"), py::arg("a"), py::arg("b"));
    distributions.def("exponential_pdf", &mathbr::distributions::exponential_pdf,
                      py::arg("x"), py::arg("rate"));
    distributions.def("exponential_cdf", &mathbr::distributions::exponential_cdf,
                      py::arg("x"), py::arg("rate"));
    distributions.def("exponential_ppf", &mathbr::distributions::exponential_ppf,
                      py::arg("p"), py::arg("rate"));
    distributions.def("weibull_pdf", &mathbr::distributions::weibull_pdf,
                      py::arg("x"), py::arg("shape"), py::arg("scale"));
    distributions.def("weibull_cdf", &mathbr::distributions::weibull_cdf,
                      py::arg("x"), py::arg("shape"), py::arg("scale"));
    distributions.def("weibull_ppf", &mathbr::distributions::weibull_ppf,
                      py::arg("p"), py::arg("shape"), py::arg("scale"));
    distributions.def("laplace_pdf", &mathbr::distributions::laplace_pdf,
                      py::arg("x"), py::arg("location"), py::arg("scale"));
    distributions.def("laplace_cdf", &mathbr::distributions::laplace_cdf,
                      py::arg("x"), py::arg("location"), py::arg("scale"));
    distributions.def("laplace_ppf", &mathbr::distributions::laplace_ppf,
                      py::arg("p"), py::arg("location"), py::arg("scale"));
    distributions.def("cauchy_pdf", &mathbr::distributions::cauchy_pdf,
                      py::arg("x"), py::arg("location"), py::arg("scale"));
    distributions.def("cauchy_cdf", &mathbr::distributions::cauchy_cdf,
                      py::arg("x"), py::arg("location"), py::arg("scale"));
    distributions.def("cauchy_ppf", &mathbr::distributions::cauchy_ppf,
                      py::arg("p"), py::arg("location"), py::arg("scale"));
    distributions.def("lognormal_pdf", &mathbr::distributions::lognormal_pdf,
                      py::arg("x"), py::arg("mu"), py::arg("sigma"));
    distributions.def("lognormal_cdf", &mathbr::distributions::lognormal_cdf,
                      py::arg("x"), py::arg("mu"), py::arg("sigma"));
    distributions.def("lognormal_ppf", &mathbr::distributions::lognormal_ppf,
                      py::arg("p"), py::arg("mu"), py::arg("sigma"));
    distributions.def("gamma_pdf", &mathbr::distributions::gamma_pdf,
                      py::arg("x"), py::arg("shape"), py::arg("scale"));
    distributions.def("gamma_cdf", &mathbr::distributions::gamma_cdf,
                      py::arg("x"), py::arg("shape"), py::arg("scale"));
    distributions.def("gamma_ppf", &mathbr::distributions::gamma_ppf,
                      py::arg("p"), py::arg("shape"), py::arg("scale"));
    distributions.def("beta_pdf", &mathbr::distributions::beta_pdf,
                      py::arg("x"), py::arg("alpha"), py::arg("beta"));
    distributions.def("beta_cdf", &mathbr::distributions::beta_cdf,
                      py::arg("x"), py::arg("alpha"), py::arg("beta"));
    distributions.def("beta_ppf", &mathbr::distributions::beta_ppf,
                      py::arg("p"), py::arg("alpha"), py::arg("beta"));
    distributions.def("pareto_pdf", &mathbr::distributions::pareto_pdf,
                      py::arg("x"), py::arg("shape"), py::arg("scale"));
    distributions.def("pareto_cdf", &mathbr::distributions::pareto_cdf,
                      py::arg("x"), py::arg("shape"), py::arg("scale"));
    distributions.def("pareto_ppf", &mathbr::distributions::pareto_ppf,
                      py::arg("p"), py::arg("shape"), py::arg("scale"));
    distributions.def("pareto_sample", &mathbr::distributions::pareto_sample,
                      py::arg("count"), py::arg("shape"), py::arg("scale"), py::arg("seed"));
    distributions.def("dirichlet_logpdf", &mathbr::distributions::dirichlet_logpdf,
                      py::arg("x"), py::arg("alpha"));
    distributions.def("dirichlet_pdf", &mathbr::distributions::dirichlet_pdf,
                      py::arg("x"), py::arg("alpha"));
    distributions.def("dirichlet_sample", &mathbr::distributions::dirichlet_sample,
                      py::arg("count"), py::arg("alpha"), py::arg("seed"));
    distributions.def("normal_fit", &mathbr::distributions::normal_fit,
                      py::arg("observations"));
    distributions.def("exponential_fit", &mathbr::distributions::exponential_fit,
                      py::arg("observations"));
    distributions.def("poisson_fit", &mathbr::distributions::poisson_fit,
                      py::arg("observations"));

    py::module_ statistics = m.def_submodule("statistics", "Descriptive statistics");
    statistics.def("mahalanobis_distance", &mathbr::multivariate::mahalanobis_distance,
                   py::arg("x"), py::arg("mean"), py::arg("covariance"));
    statistics.def("mean", &mathbr::statistics::mean, py::arg("x"));
    statistics.def("median", &mathbr::statistics::median, py::arg("x"));
    statistics.def("mode", &mathbr::statistics::mode, py::arg("x"));
    statistics.def("variance", &mathbr::statistics::variance,
                   py::arg("x"), py::arg("ddof") = 1);
    statistics.def("standard_deviation", &mathbr::statistics::standard_deviation,
                   py::arg("x"), py::arg("ddof") = 1);
    statistics.def("skewness", &mathbr::statistics::skewness, py::arg("x"));
    statistics.def("excess_kurtosis", &mathbr::statistics::excess_kurtosis, py::arg("x"));
    statistics.def("quantile", &mathbr::statistics::quantile,
                   py::arg("x"), py::arg("p"));
    statistics.def("percentile", &mathbr::statistics::percentile,
                   py::arg("x"), py::arg("p"));
    statistics.def("interquartile_range", &mathbr::statistics::interquartile_range, py::arg("x"));
    statistics.def("five_number_summary", &mathbr::statistics::five_number_summary, py::arg("x"));
    statistics.def("covariance", &mathbr::statistics::covariance,
                   py::arg("x"), py::arg("y"), py::arg("ddof") = 1);
    statistics.def("pearson_correlation", &mathbr::statistics::pearson_correlation,
                   py::arg("x"), py::arg("y"));
    statistics.def("spearman_correlation", &mathbr::statistics::spearman_correlation,
                  py::arg("x"), py::arg("y"));
    statistics.def("kendall_tau", &mathbr::statistics::kendall_tau,
                   py::arg("x"), py::arg("y"));
    statistics.def("covariance_matrix", &mathbr::statistics::covariance_matrix,
                   py::arg("data"), py::arg("ddof") = 1);
    statistics.def("correlation_matrix", &mathbr::statistics::correlation_matrix,
                   py::arg("data"));
    statistics.def("spearman_correlation_matrix", &mathbr::statistics::spearman_correlation_matrix,
                   py::arg("data"));
    statistics.def("kendall_correlation_matrix", &mathbr::statistics::kendall_correlation_matrix,
                   py::arg("data"));
    statistics.def("weighted_covariance_matrix", &mathbr::statistics::weighted_covariance_matrix,
                   py::arg("data"), py::arg("weights"));
    statistics.def("weighted_correlation_matrix", &mathbr::statistics::weighted_correlation_matrix,
                   py::arg("data"), py::arg("weights"));
    statistics.def("weighted_mean", &mathbr::statistics::weighted_mean,
                   py::arg("x"), py::arg("weights"));
    statistics.def("weighted_variance", &mathbr::statistics::weighted_variance,
                   py::arg("x"), py::arg("weights"));
    statistics.def("weighted_covariance", &mathbr::statistics::weighted_covariance,
                   py::arg("x"), py::arg("y"), py::arg("weights"));
    statistics.def("weighted_correlation", &mathbr::statistics::weighted_correlation,
                   py::arg("x"), py::arg("y"), py::arg("weights"));
    statistics.def("weighted_quantile", &mathbr::statistics::weighted_quantile,
                   py::arg("x"), py::arg("weights"), py::arg("p"));
    statistics.def("log_sum_exp", &mathbr::statistics::log_sum_exp, py::arg("x"));
    statistics.def("log_empirical_mgf", &mathbr::statistics::log_empirical_mgf,
                   py::arg("x"), py::arg("t"));
    statistics.def("empirical_mgf", &mathbr::statistics::empirical_mgf,
                   py::arg("x"), py::arg("t"));

    py::module_ hypothesis = m.def_submodule("hypothesis", "Classical hypothesis tests");
    py::class_<mathbr::hypothesis::TTestResult>(hypothesis, "TTestResult")
        .def_readonly("statistic", &mathbr::hypothesis::TTestResult::statistic)
        .def_readonly("degrees_of_freedom", &mathbr::hypothesis::TTestResult::degrees_of_freedom)
        .def_readonly("p_value", &mathbr::hypothesis::TTestResult::p_value);
    py::class_<mathbr::hypothesis::ZTestResult>(hypothesis, "ZTestResult")
        .def_readonly("statistic", &mathbr::hypothesis::ZTestResult::statistic)
        .def_readonly("p_value", &mathbr::hypothesis::ZTestResult::p_value);
    py::class_<mathbr::hypothesis::ChiSquareTestResult>(hypothesis, "ChiSquareTestResult")
        .def_readonly("statistic", &mathbr::hypothesis::ChiSquareTestResult::statistic)
        .def_readonly("degrees_of_freedom",
                      &mathbr::hypothesis::ChiSquareTestResult::degrees_of_freedom)
        .def_readonly("p_value", &mathbr::hypothesis::ChiSquareTestResult::p_value);
    py::class_<mathbr::hypothesis::AnovaResult>(hypothesis, "AnovaResult")
        .def_readonly("statistic", &mathbr::hypothesis::AnovaResult::statistic)
        .def_readonly("df_between", &mathbr::hypothesis::AnovaResult::df_between)
        .def_readonly("df_within", &mathbr::hypothesis::AnovaResult::df_within)
        .def_readonly("p_value", &mathbr::hypothesis::AnovaResult::p_value);
    py::class_<mathbr::hypothesis::MannWhitneyResult>(hypothesis, "MannWhitneyResult")
        .def_readonly("statistic", &mathbr::hypothesis::MannWhitneyResult::statistic)
        .def_readonly("z_score", &mathbr::hypothesis::MannWhitneyResult::z_score)
        .def_readonly("p_value", &mathbr::hypothesis::MannWhitneyResult::p_value);
    hypothesis.def("one_sample_t_test", &mathbr::hypothesis::one_sample_t_test,
                   py::arg("x"), py::arg("null_mean") = 0.0);
    hypothesis.def("paired_t_test", &mathbr::hypothesis::paired_t_test,
                   py::arg("before"), py::arg("after"));
    hypothesis.def("welch_t_test", &mathbr::hypothesis::welch_t_test,
                   py::arg("x"), py::arg("y"));
    hypothesis.def("bonferroni_correction", &mathbr::hypothesis::bonferroni_correction,
                   py::arg("p_values"));
    hypothesis.def("holm_correction", &mathbr::hypothesis::holm_correction,
                   py::arg("p_values"));
    hypothesis.def("benjamini_hochberg_correction",
                   &mathbr::hypothesis::benjamini_hochberg_correction,
                   py::arg("p_values"));
    hypothesis.def("proportion_z_test", &mathbr::hypothesis::proportion_z_test,
                   py::arg("successes"), py::arg("trials"), py::arg("null_p"));
    hypothesis.def("chi_square_goodness_of_fit",
                   &mathbr::hypothesis::chi_square_goodness_of_fit,
                   py::arg("observed"), py::arg("expected"));
    hypothesis.def("one_way_anova", &mathbr::hypothesis::one_way_anova,
                   py::arg("groups"));
    hypothesis.def("kruskal_wallis", &mathbr::hypothesis::kruskal_wallis,
                   py::arg("groups"));
    hypothesis.def("chi_square_independence", &mathbr::hypothesis::chi_square_independence,
                   py::arg("table"));
    hypothesis.def("mann_whitney_u", &mathbr::hypothesis::mann_whitney_u,
                   py::arg("x"), py::arg("y"));

    py::module_ evaluation = m.def_submodule("evaluation", "Binary classification evaluation");
    py::class_<mathbr::evaluation::RocCurve>(evaluation, "RocCurve")
        .def_readonly("fpr", &mathbr::evaluation::RocCurve::fpr)
        .def_readonly("tpr", &mathbr::evaluation::RocCurve::tpr)
        .def_readonly("thresholds", &mathbr::evaluation::RocCurve::thresholds);
    py::class_<mathbr::evaluation::PrecisionRecallCurve>(evaluation, "PrecisionRecallCurve")
        .def_readonly("precision", &mathbr::evaluation::PrecisionRecallCurve::precision)
        .def_readonly("recall", &mathbr::evaluation::PrecisionRecallCurve::recall)
        .def_readonly("thresholds", &mathbr::evaluation::PrecisionRecallCurve::thresholds);
    py::class_<mathbr::evaluation::CalibrationCurve>(evaluation, "CalibrationCurve")
        .def_readonly("mean_predicted", &mathbr::evaluation::CalibrationCurve::mean_predicted)
        .def_readonly("fraction_positive", &mathbr::evaluation::CalibrationCurve::fraction_positive)
        .def_readonly("counts", &mathbr::evaluation::CalibrationCurve::counts);
    evaluation.def("confusion_matrix", &mathbr::evaluation::confusion_matrix,
                   py::arg("y_true"), py::arg("y_pred"));
    evaluation.def("precision", &mathbr::evaluation::precision,
                   py::arg("y_true"), py::arg("y_pred"));
    evaluation.def("recall", &mathbr::evaluation::recall,
                   py::arg("y_true"), py::arg("y_pred"));
    evaluation.def("f1_score", &mathbr::evaluation::f1_score,
                   py::arg("y_true"), py::arg("y_pred"));
    evaluation.def("roc_curve", &mathbr::evaluation::roc_curve,
                   py::arg("y_true"), py::arg("scores"));
    evaluation.def("roc_auc", &mathbr::evaluation::roc_auc,
                   py::arg("y_true"), py::arg("scores"));
    evaluation.def("precision_recall_curve", &mathbr::evaluation::precision_recall_curve,
                   py::arg("y_true"), py::arg("scores"));
    evaluation.def("rmse", &mathbr::evaluation::rmse, py::arg("y_true"), py::arg("y_pred"));
    evaluation.def("mae", &mathbr::evaluation::mae, py::arg("y_true"), py::arg("y_pred"));
    evaluation.def("mape", &mathbr::evaluation::mape, py::arg("y_true"), py::arg("y_pred"));
    evaluation.def("log_loss", &mathbr::evaluation::log_loss,
                   py::arg("y_true"), py::arg("probabilities"));
    evaluation.def("k_fold_indices", &mathbr::evaluation::k_fold_indices,
                   py::arg("n_samples"), py::arg("n_splits"), py::arg("seed") = 0);
    evaluation.def("calibration_curve", &mathbr::evaluation::calibration_curve,
                   py::arg("y_true"), py::arg("probabilities"), py::arg("n_bins") = 10);

    py::module_ diagnostics = m.def_submodule("diagnostics", "Regression diagnostics and information criteria");
    diagnostics.def("aic", &mathbr::diagnostics::aic,
                    py::arg("log_likelihood"), py::arg("n_parameters"));
    diagnostics.def("bic", &mathbr::diagnostics::bic,
                    py::arg("log_likelihood"), py::arg("n_parameters"), py::arg("n_observations"));
    diagnostics.def("hqic", &mathbr::diagnostics::hqic,
                    py::arg("log_likelihood"), py::arg("n_parameters"), py::arg("n_observations"));
    diagnostics.def("residual_standard_error", &mathbr::diagnostics::residual_standard_error,
                    py::arg("residuals"), py::arg("n_parameters"));
    diagnostics.def("durbin_watson", &mathbr::diagnostics::durbin_watson,
                    py::arg("residuals"));
    diagnostics.def("jarque_bera_statistic", &mathbr::diagnostics::jarque_bera_statistic,
                    py::arg("residuals"));
    diagnostics.def("jarque_bera_p_value", &mathbr::diagnostics::jarque_bera_p_value,
                    py::arg("residuals"));

    py::module_ nonparametric = m.def_submodule("nonparametric", "Nonparametric estimation");
    nonparametric.def("empirical_cdf", &mathbr::nonparametric::empirical_cdf,
                      py::arg("sample"), py::arg("points"));
    nonparametric.def("gaussian_kde", &mathbr::nonparametric::gaussian_kde,
                      py::arg("sample"), py::arg("points"), py::arg("bandwidth"));
    nonparametric.def("nadaraya_watson", &mathbr::nonparametric::nadaraya_watson,
                      py::arg("x"), py::arg("y"), py::arg("points"), py::arg("bandwidth"));

    py::module_ time_series_diagnostics = m.def_submodule("time_series_diagnostics", "Time-series diagnostics");
    py::class_<mathbr::time_series_diagnostics::LjungBoxResult>(time_series_diagnostics, "LjungBoxResult")
        .def_readonly("statistic", &mathbr::time_series_diagnostics::LjungBoxResult::statistic)
        .def_readonly("p_value", &mathbr::time_series_diagnostics::LjungBoxResult::p_value)
        .def_readonly("lags", &mathbr::time_series_diagnostics::LjungBoxResult::lags);
    time_series_diagnostics.def("acf", &mathbr::time_series_diagnostics::acf,
                                py::arg("x"), py::arg("max_lag"));
    time_series_diagnostics.def("pacf", &mathbr::time_series_diagnostics::pacf,
                                py::arg("x"), py::arg("max_lag"));
    time_series_diagnostics.def("ljung_box", &mathbr::time_series_diagnostics::ljung_box,
                                py::arg("x"), py::arg("lags"));

    py::module_ survival = m.def_submodule("survival", "Right-censored survival analysis");
    py::class_<mathbr::survival::KaplanMeierResult>(survival, "KaplanMeierResult")
        .def_readonly("times", &mathbr::survival::KaplanMeierResult::times)
        .def_readonly("survival", &mathbr::survival::KaplanMeierResult::survival)
        .def_readonly("at_risk", &mathbr::survival::KaplanMeierResult::at_risk)
        .def_readonly("events", &mathbr::survival::KaplanMeierResult::events);
    py::class_<mathbr::survival::LogRankResult>(survival, "LogRankResult")
        .def_readonly("statistic", &mathbr::survival::LogRankResult::statistic)
        .def_readonly("p_value", &mathbr::survival::LogRankResult::p_value);
    survival.def("kaplan_meier", &mathbr::survival::kaplan_meier,
                 py::arg("times"), py::arg("events"));
    survival.def("log_rank_test", &mathbr::survival::log_rank_test,
                 py::arg("times"), py::arg("events"), py::arg("groups"));

    py::module_ bayesian = m.def_submodule("bayesian", "Conjugate Bayesian updating");
    py::class_<mathbr::bayesian::BetaPosterior>(bayesian, "BetaPosterior")
        .def_readonly("alpha", &mathbr::bayesian::BetaPosterior::alpha)
        .def_readonly("beta", &mathbr::bayesian::BetaPosterior::beta)
        .def("mean", &mathbr::bayesian::BetaPosterior::mean)
        .def("credible_interval", &mathbr::bayesian::BetaPosterior::credible_interval,
             py::arg("level") = 0.95);
    py::class_<mathbr::bayesian::NormalPosterior>(bayesian, "NormalPosterior")
        .def_readonly("mean", &mathbr::bayesian::NormalPosterior::mean)
        .def_readonly("standard_deviation", &mathbr::bayesian::NormalPosterior::standard_deviation)
        .def("credible_interval", &mathbr::bayesian::NormalPosterior::credible_interval,
             py::arg("level") = 0.95);
    bayesian.def("beta_binomial_update", &mathbr::bayesian::beta_binomial_update,
                 py::arg("alpha"), py::arg("beta"), py::arg("successes"), py::arg("trials"));
    bayesian.def("normal_normal_update", &mathbr::bayesian::normal_normal_update,
                 py::arg("prior_mean"), py::arg("prior_sd"),
                 py::arg("observation_sd"), py::arg("observations"));

    // activations
    py::module_ activations = m.def_submodule("activations", "Activation functions");
    activations.def("sigmoid", &Activations::sigmoid, py::arg("z"));
    activations.def("sigmoid_derivative", &Activations::sigmoid_derivative, py::arg("z"));
    activations.def("tanh_activation", &Activations::tanh_activation, py::arg("z"));
    activations.def("tanh_derivative", &Activations::tanh_derivative, py::arg("z"));
    activations.def("relu", &Activations::relu, py::arg("z"));
    activations.def("relu_derivative", &Activations::relu_derivative, py::arg("z"));
    activations.def("leaky_relu", &Activations::leaky_relu, py::arg("z"), py::arg("alpha") = 0.01);
    activations.def("leaky_relu_derivative", &Activations::leaky_relu_derivative, py::arg("z"), py::arg("alpha") = 0.01);
    activations.def("gelu", &Activations::gelu, py::arg("z"));
    activations.def("gelu_derivative", &Activations::gelu_derivative, py::arg("z"));
    activations.def("swish", &Activations::swish, py::arg("z"));
    activations.def("swish_derivative", &Activations::swish_derivative, py::arg("z"));
    activations.def("softmax", &Activations::softmax, py::arg("z"));

    // losses
    py::module_ losses = m.def_submodule("losses", "Loss functions and metrics");
    losses.def("mse", &Losses::mse, py::arg("y_true"), py::arg("y_pred"));
    losses.def("mse_derivative", &Losses::mse_derivative, py::arg("y_true"), py::arg("y_pred"));
    losses.def("mae", &Losses::mae, py::arg("y_true"), py::arg("y_pred"));
    losses.def("rmse", &Losses::rmse, py::arg("y_true"), py::arg("y_pred"));
    losses.def("logloss", &Losses::logloss, py::arg("y_true"), py::arg("y_pred"));

    // linear regression
    py::class_<LinearRegression>(m, "LinearRegression")
        .def(py::init<int>(), py::arg("n_features"))
        .def("predict", &LinearRegression::predict, py::arg("x"))
        .def("predict_batch", &LinearRegression::predict_batch, py::arg("X"))
        .def("fit", &LinearRegression::fit, py::arg("X"), py::arg("y"),
             py::arg("lr") = 0.01, py::arg("epochs") = 1000)
        .def("get_weights", &LinearRegression::get_weights)
        .def("get_bias", &LinearRegression::get_bias)
        .def("trained", &LinearRegression::trained)
        .def("__repr__", [](const LinearRegression& r) {
            return "<LinearRegression trained=" + std::string(r.trained() ? "True" : "False") + ">";
        });

    // logistic regression
    py::class_<LogisticRegression>(m, "LogisticRegression")
        .def(py::init<int>(), py::arg("n_features"))
        .def("predict", &LogisticRegression::predict, py::arg("x"), py::arg("threshold") = 0.5)
        .def("predict_proba", &LogisticRegression::predict_proba, py::arg("x"))
        .def("predict_proba_batch", &LogisticRegression::predict_proba_batch, py::arg("X"))
        .def("fit", &LogisticRegression::fit, py::arg("X"), py::arg("y"),
             py::arg("lr") = 0.01, py::arg("epochs") = 1000)
        .def("get_weights", &LogisticRegression::get_weights)
        .def("get_bias", &LogisticRegression::get_bias)
        .def("trained", &LogisticRegression::trained)
        .def("__repr__", [](const LogisticRegression& r) {
            return "<LogisticRegression trained=" + std::string(r.trained() ? "True" : "False") + ">";
        });

    py::class_<mathbr::OLS>(m, "OLS", "Ordinary least squares with an intercept and classical standard errors.")
        .def(py::init<int>(), py::arg("n_features"))
        .def("fit", &mathbr::OLS::fit, py::arg("X"), py::arg("y"),
             "Fit by QR decomposition. Requires more observations than parameters and full column rank.")
        .def("predict", &mathbr::OLS::predict, py::arg("x"))
        .def("predict_batch", &mathbr::OLS::predict_batch, py::arg("X"))
        .def("coefficients", &mathbr::OLS::coefficients, "Intercept first, followed by feature coefficients.")
        .def("standard_errors", &mathbr::OLS::standard_errors,
             "Classical OLS standard errors under homoscedastic, independent errors.")
        .def("t_statistics", &mathbr::OLS::t_statistics)
        .def("p_values", &mathbr::OLS::p_values)
        .def("confidence_intervals", &mathbr::OLS::confidence_intervals,
             py::arg("level") = 0.95)
        .def("r_squared", &mathbr::OLS::r_squared)
        .def("adjusted_r_squared", &mathbr::OLS::adjusted_r_squared)
        .def("f_statistic", &mathbr::OLS::f_statistic)
        .def("residual_variance", &mathbr::OLS::residual_variance)
        .def("degrees_of_freedom", &mathbr::OLS::degrees_of_freedom)
        .def("trained", &mathbr::OLS::trained);

    py::class_<mathbr::WLS, mathbr::OLS>(m, "WLS", "Weighted least squares with positive inverse-variance weights.")
        .def(py::init<int>(), py::arg("n_features"))
        .def("fit", &mathbr::WLS::fit, py::arg("X"), py::arg("y"), py::arg("weights"),
             "Fit weighted least squares. Weights must be positive and finite.");

    py::class_<mathbr::RegularizedRegression>(m, "_RegularizedRegression")
        .def("fit", &mathbr::RegularizedRegression::fit, py::arg("X"), py::arg("y"))
        .def("predict", &mathbr::RegularizedRegression::predict, py::arg("x"))
        .def("predict_batch", &mathbr::RegularizedRegression::predict_batch, py::arg("X"))
        .def("coefficients", &mathbr::RegularizedRegression::coefficients)
        .def("intercept", &mathbr::RegularizedRegression::intercept)
        .def("iterations", &mathbr::RegularizedRegression::iterations)
        .def("converged", &mathbr::RegularizedRegression::converged)
        .def("trained", &mathbr::RegularizedRegression::trained);
    py::class_<mathbr::Ridge, mathbr::RegularizedRegression>(m, "Ridge")
        .def(py::init<int, double, int, double>(), py::arg("n_features"), py::arg("alpha") = 1.0,
             py::arg("max_iter") = 1000, py::arg("tol") = 1e-8);
    py::class_<mathbr::Lasso, mathbr::RegularizedRegression>(m, "Lasso")
        .def(py::init<int, double, int, double>(), py::arg("n_features"), py::arg("alpha") = 1.0,
             py::arg("max_iter") = 1000, py::arg("tol") = 1e-8);
    py::class_<mathbr::ElasticNet, mathbr::RegularizedRegression>(m, "ElasticNet")
        .def(py::init<int, double, double, int, double>(), py::arg("n_features"),
             py::arg("alpha") = 1.0, py::arg("l1_ratio") = 0.5,
             py::arg("max_iter") = 1000, py::arg("tol") = 1e-8);

    py::class_<mathbr::IV2SLS>(m, "IV2SLS", "Two-stage least squares with excluded instruments.")
        .def(py::init<int, int, int>(), py::arg("n_exog"), py::arg("n_endog"), py::arg("n_instruments"))
        .def("fit", &mathbr::IV2SLS::fit, py::arg("exog"), py::arg("endog"),
             py::arg("instruments"), py::arg("y"))
        .def("predict", &mathbr::IV2SLS::predict, py::arg("exog"), py::arg("endog"))
        .def("coefficients", &mathbr::IV2SLS::coefficients,
             "Intercept, exogenous coefficients, then endogenous coefficients.")
        .def("first_stage_r_squared", &mathbr::IV2SLS::first_stage_r_squared,
             "Overall first-stage R-squared for each endogenous regressor; not a weak-instrument test.")
        .def("trained", &mathbr::IV2SLS::trained);

    py::class_<mathbr::FixedEffects>(m, "FixedEffects", "Entity fixed-effects panel regression.")
        .def(py::init<int>(), py::arg("n_features"))
        .def("fit", &mathbr::FixedEffects::fit, py::arg("X"), py::arg("y"), py::arg("entity_ids"))
        .def("predict", &mathbr::FixedEffects::predict, py::arg("x"), py::arg("entity_id"))
        .def("coefficients", &mathbr::FixedEffects::coefficients, "Within-entity slopes.")
        .def("entity_intercept", &mathbr::FixedEffects::entity_intercept, py::arg("entity_id"))
        .def("within_r_squared", &mathbr::FixedEffects::within_r_squared)
        .def("trained", &mathbr::FixedEffects::trained);

    py::class_<mathbr::VAR>(m, "VAR", "Vector autoregression with an intercept.")
        .def(py::init<int, int>(), py::arg("n_series"), py::arg("lags"))
        .def("fit", &mathbr::VAR::fit, py::arg("observations"))
        .def("coefficients", &mathbr::VAR::coefficients,
             "One row per equation: intercept, then lag-1 series, lag-2 series, etc.")
        .def("forecast", &mathbr::VAR::forecast, py::arg("steps"))
        .def("residual_covariance", &mathbr::VAR::residual_covariance)
        .def("trained", &mathbr::VAR::trained);

    py::class_<mathbr::AR>(m, "AR", "Univariate autoregression with an intercept.")
        .def(py::init<int>(), py::arg("lags"))
        .def("fit", &mathbr::AR::fit, py::arg("observations"))
        .def("coefficients", &mathbr::AR::coefficients)
        .def("forecast", &mathbr::AR::forecast, py::arg("steps"))
        .def("trained", &mathbr::AR::trained);

    py::class_<mathbr::GARCH>(m, "GARCH", "Gaussian quasi-maximum-likelihood GARCH(1,1) with constant mean.")
        .def(py::init<int, double, bool>(), py::arg("max_iter") = 2000,
             py::arg("tol") = 1e-7, py::arg("arch_only") = false)
        .def("fit", &mathbr::GARCH::fit, py::arg("observations"))
        .def("mean", &mathbr::GARCH::mean)
        .def("omega", &mathbr::GARCH::omega)
        .def("alpha", &mathbr::GARCH::alpha)
        .def("beta", &mathbr::GARCH::beta)
        .def("log_likelihood", &mathbr::GARCH::log_likelihood)
        .def("conditional_variance", &mathbr::GARCH::conditional_variance)
        .def("forecast_variance", &mathbr::GARCH::forecast_variance, py::arg("steps"))
        .def("converged", &mathbr::GARCH::converged)
        .def("trained", &mathbr::GARCH::trained);
    py::class_<mathbr::ARCH, mathbr::GARCH>(m, "ARCH", "Gaussian ARCH(1) with constant mean.")
        .def(py::init<int, double>(), py::arg("max_iter") = 2000, py::arg("tol") = 1e-7);
}
