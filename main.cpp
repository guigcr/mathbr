#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace py = pybind11;

// activation functions
namespace Activations {

    double sigmoid(double z) {
        return 1.0 / (1.0 + std::exp(-z));
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
        const double c = std::sqrt(2.0 / M_PI);
        return 0.5 * z * (1.0 + std::tanh(c * (z + 0.044715 * std::pow(z, 3))));
    }

    double gelu_derivative(double z) {
        const double c = std::sqrt(2.0 / M_PI);
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
        double total = 0.0;
        int n = y_true.size();
        for (int i = 0; i < n; i++) {
            total += std::pow(y_true[i] - y_pred[i], 2);
        }
        return total / n;
    }

    std::vector<double> mse_derivative(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
        int n = y_true.size();
        std::vector<double> grad(n);
        for (int i = 0; i < n; i++) {
            grad[i] = 2.0 * (y_pred[i] - y_true[i]) / n;
        }
        return grad;
    }

    double mae(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
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
        double loss = 0.0;
        int n = y_true.size();
        for (int i = 0; i < n; i++) {
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
    LinearRegression(int n_features) : w(n_features, 0.0), b(0.0) {}

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

        int n = X.size();
        int n_features = w.size();

        for (int epoch = 0; epoch < epochs; epoch++) {
            std::vector<double> y_pred = predict_batch(X);
            std::vector<double> output_grad = Losses::mse_derivative(y, y_pred);

            std::vector<double> grad_w(n_features, 0.0);
            double grad_b = 0.0;

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n_features; j++) {
                    grad_w[j] += output_grad[i] * X[i][j];
                }
                grad_b += output_grad[i];
            }

            for (int j = 0; j < n_features; j++) {
                w[j] -= lr * grad_w[j] / n;
            }
            b -= lr * grad_b / n;
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
    LogisticRegression(int n_features) : w(n_features, 0.0), b(0.0) {}

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

        int n = X.size();
        int n_features = w.size();

        for (int epoch = 0; epoch < epochs; epoch++) {
            std::vector<double> y_pred = predict_proba_batch(X);

            std::vector<double> grad_w(n_features, 0.0);
            double grad_b = 0.0;

            for (int i = 0; i < n; i++) {
                double error = y_pred[i] - y[i];
                for (int j = 0; j < n_features; j++) {
                    grad_w[j] += error * X[i][j];
                }
                grad_b += error;
            }

            for (int j = 0; j < n_features; j++) {
                w[j] -= lr * grad_w[j] / n;
            }
            b -= lr * grad_b / n;
        }
        is_trained = true;
    }

    std::vector<double> get_weights() const { return w; }
    double get_bias() const { return b; }
    bool trained() const { return is_trained; }
};


PYBIND11_MODULE(mathbr, m) {
    m.doc() = "ML math library - Version 2.0";
    m.attr("version") = "0.3.0";

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
}
