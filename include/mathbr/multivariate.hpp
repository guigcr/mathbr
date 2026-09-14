#pragma once

#include <cstdint>
#include <vector>

namespace mathbr::multivariate {

class MultivariateNormal {
public:
    MultivariateNormal(const std::vector<double>& mean,
                       const std::vector<std::vector<double>>& covariance);
    double logpdf(const std::vector<double>& x) const;
    double pdf(const std::vector<double>& x) const;
    double mahalanobis_distance(const std::vector<double>& x) const;
    std::vector<double> logpdf_batch(const std::vector<std::vector<double>>& data) const;
    std::vector<double> pdf_batch(const std::vector<std::vector<double>>& data) const;
    std::vector<std::vector<double>> sample(std::size_t count, std::uint64_t seed) const;

private:
    double squared_distance(const std::vector<double>& x) const;
    double squared_distance(const std::vector<double>& x,
                            std::vector<double>& scratch) const;
    std::vector<double> mean_;
    std::vector<std::vector<double>> lower_;
    double log_normalizer_;
};

double mahalanobis_distance(const std::vector<double>& x,
                            const std::vector<double>& mean,
                            const std::vector<std::vector<double>>& covariance);

}  // namespace mathbr::multivariate
