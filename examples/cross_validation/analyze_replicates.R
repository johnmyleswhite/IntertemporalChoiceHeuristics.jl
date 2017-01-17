library("ggplot2")
library("dplyr")

cv <- read.csv(
    file.path("examples", "cross_validation", "replicates.tsv"),
    sep = "\t",
    header = FALSE,
    stringsAsFactors = FALSE
)

names(cv) <- c(
    "iteration",
    "model",
    "training_loss",
    "lambda",
    "training_proportion",
    "evaluation_loss",
    "value"
)

cv <- transform(cv, lambda = factor(lambda))
cv <- transform(cv, training_proportion = factor(training_proportion))

perf <- cv %>%
    group_by(
        training_proportion,
        training_loss,
        lambda,
        evaluation_loss,
        model
    ) %>%
    summarize(
        count = n(),
        failure_rate = mean(100 * is.na(value)),
        mean = mean(value, na.rm = TRUE),
        median = quantile(value, 0.50, na.rm = TRUE),
        lower = quantile(value, 0.025, na.rm = TRUE),
        upper = quantile(value, 0.975, na.rm = TRUE)
    ) %>%
    arrange(training_proportion, training_loss, evaluation_loss, mean)

write.csv(
    perf,
    file.path("examples", "cross_validation", "performance.csv"),
    row.names = FALSE,
)

p <- ggplot(
        perf,
        aes(
            x = median,
            y = mean,
            color = evaluation_loss,
            shape = lambda
        )
    ) +
    geom_point() +
    geom_abline() +
    facet_grid(training_proportion ~ training_loss) +
    xlab("Median Value of Loss Function for Model") +
    ylab("Mean Value of Loss Function for Model") +
    ggtitle("Log Loss Function Behaves Erratically in Small Samples") +
    theme_bw()

ggsave(
    file.path(
        "examples",
        "cross_validation",
        "graphs",
        "mean_vs_median_loss.pdf"
    ),
    height = 8,
    width = 12
)

for (my_training_proportion in unique(cv$training_proportion)) {
    for (my_evaluation_loss in unique(cv$evaluation_loss)) {
        p <- ggplot(
            cv %>%
                filter(
                    training_proportion == my_training_proportion,
                    evaluation_loss == my_evaluation_loss
                ),
            aes(x = model, y = value, color = model)
        ) +
            geom_point() +
            facet_grid(training_loss ~ lambda) +
            coord_flip() +
            xlab("Model") +
            ylab("Out-of-Sample Loss Value") +
            ggtitle(
                paste0(
                    my_evaluation_loss,
                    ", Training Proportion = ",
                    my_training_proportion
                )
            ) +
            theme_bw() +
            theme(legend.position = "none")

        dir_path <- file.path(
            "examples",
            "cross_validation",
            "graphs",
            gsub("\\W", "_", my_evaluation_loss)
        )

        dir.create(dir_path, showWarnings = FALSE)

        ggsave(
            file.path(
                dir_path,
                paste0(my_training_proportion, "_scatterplot.pdf")
            ),
            height = 16,
            width = 24
        )

        p <- ggplot(
            perf %>%
                filter(
                    training_proportion == my_training_proportion,
                    evaluation_loss == my_evaluation_loss
                ),
            aes(
                x = model,
                y = median,
                ymin = lower,
                ymax = upper,
                color = model
            )
        ) +
            geom_point() +
            geom_point(aes(y = mean), size = 2.5) +
            geom_errorbar() +
            facet_grid(training_loss ~ lambda) +
            coord_flip() +
            xlab("Model") +
            ylab("Out-of-Sample Loss Value") +
            ggtitle(
                paste0(
                    my_evaluation_loss,
                    ", Training Proportion = ",
                    my_training_proportion
                )
            ) +
            theme_bw() +
            theme(legend.position = "none")

        ggsave(
            file.path(
                dir_path,
                paste0(my_training_proportion, "_quantiles.pdf")
            ),
            height = 16,
            width = 24
        )
    }
}
