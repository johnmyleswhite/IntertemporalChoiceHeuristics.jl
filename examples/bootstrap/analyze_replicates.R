library("ggplot2")
library("GGally")
library("dplyr")

replicates <- read.csv(
    file.path("examples", "bootstrap", "replicates.tsv"),
    sep = "\t",
    header = FALSE,
    stringsAsFactors = FALSE
)

names(replicates) <- c(
    "replicate",
    "model",
    "theta1",
    "theta2",
    "theta3",
    "theta4",
    "theta5"
)

grid.draw.gg <- function(x) {
    return(print(x))
}

for (my_model in unique(replicates$model)) {
    p <- ggpairs(
        replicates %>% filter(model == my_model),
        columns = c("theta1", "theta2", "theta3", "theta4", "theta5")
    ) +
        ggtitle(paste(my_model, "Pairwise Parameter Sampling Distributions"))

    # TODO: Diagnose plotting failure that happens for one model here.
    # Given the inscrutable error message, it could be a ggplot2 bug.
    try(
        ggsave(
            file.path(
                "examples",
                "bootstrap",
                "graphs",
                paste0(my_model, ".pdf")
            ),
            p,
            height = 16,
            width = 16
        )
    )
}
