library("ggplot2")
library("dplyr")

estimates <- read.csv(
    file.path("examples", "MLE", "estimates.tsv"),
    sep = "\t",
    header = FALSE,
    stringsAsFactors = FALSE
)

names(estimates) <- c("model", "evaluation_loss", "value", "nll", "norm")

write.csv(
    estimates %>% arrange(evaluation_loss, value),
    file = file.path("examples", "MLE", "performance.csv"),
    row.names = FALSE
)
