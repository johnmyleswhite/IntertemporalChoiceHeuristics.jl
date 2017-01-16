library("ggplot2")
library("dplyr")

grid <- read.csv(
    file.path("examples", "profile_likelihood", "grid.tsv"),
    sep = "\t",
    header = FALSE,
    stringsAsFactors = FALSE
)

names(grid) <- c(
    "model", "parameter", "parameter_value", "nll", "mle", "se"
)

for (my_model in unique(grid$model)) {
    model_grid <- grid %>% filter(model == my_model)
    for (j in unique(model_grid$parameter)) {
        p <- ggplot(
            model_grid %>% filter(parameter == j),
            aes(x = parameter_value, y = nll)
        ) +
            geom_line() +
            theme_bw() +
            ggtitle(
                paste(my_model, "Profile Likelihood for Parameter", j)
            ) +
            xlab("Parameter Value") +
            ylab("Negative Log Likelihood")

        ggsave(
            file.path(
                "examples",
                "profile_likelihood",
                "graphs",
                paste0(my_model, "_", j, ".pdf")
            ),
            p,
            height = 8,
            width = 8
        )
    }
}
