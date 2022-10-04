import igraph as ig
import statsmodels.api as sm
import numpy as np
from inequality.theil import TheilD
from .base_reward import BaseReward
import logging

logger = logging.getLogger(__name__)


class EgalitarianJSDReward(BaseReward):
    """
    Attention: This reward is still not fully functional. Here, we're trying
    """

    def _reward_scaling(self, reward: float) -> float:
        return -reward

    def _evaluate(self, g: ig.Graph, *args, **kwargs) -> float:
        logger.warning("The EgalitarianJSDReward is not fully functional! Use at own risk!")
        metrics_dfs = self.retrieve_dfs(g)

        # fit KDE (sklearn) on each component
        kdes = {group: {metric: None for metric in self.metrics_names} for group in self.groups}
        kde_mixtures = {metric: None for metric in self.metrics_names}

        for metric, metric_df in metrics_dfs.items():
            # fig, ax = plt.subplots()
            # fig.suptitle(f"Plot for {metric=}")
            for group in self.groups:
                X = metric_df[metric_df.group == group].drop(columns='group').astype(float).to_numpy()
                kde = sm.nonparametric.KDEUnivariate(X)
                kde.fit(bw=0.2)
                kdes[group][metric] = kde
            #     # score_samples returns the log of the probability density
            #     ax.plot(kde.support, kde.density, lw=3, label=f"KDE from samples {group=}", zorder=10, color=group)
            #     ax.scatter(
            #         X,
            #         np.abs(np.random.randn(X.size)) / 40,
            #         marker="x",
            #         color=group,
            #         zorder=20,
            #         label=f"Samples {group=}",
            #         alpha=0.5,
            #     )
            #     ax.legend(loc="best")
            #     ax.grid(True, zorder=-5)
            # plt.show()

        for metric, metric_df in metrics_dfs.items():
            X = metric_df.drop(columns='group').astype(float).to_numpy()
            kde = sm.nonparametric.KDEUnivariate(X)
            kde.fit(bw=0.2)
            kde_mixtures[metric] = kde

        reward = 0
        for metric in metrics_dfs:
            n_dist = len(kdes.keys())
            reward += kde_mixtures[metric].entropy - 1 / n_dist * sum([kdes[group][metric].entropy for group in kdes])

        return reward


class EgalitarianTheilReward(BaseReward):
    """

    """

    def _evaluate(self, g: ig.Graph, *args, **kwargs) -> float:
        metrics_dfs = self.retrieve_dfs(g)

        theil_inequality = {metric: None for metric in self.metrics_names}

        for metric, metric_df in metrics_dfs.items():
            X = metric_df.drop(columns='group').astype(float).to_numpy()
            Y = metric_df.group
            theil_t = TheilD(X, Y).T[0] if X.sum() > 0 else 0.0
            theil_inequality[metric] = theil_t

        return sum(theil_inequality.values())

    def _reward_scaling(self, reward: float) -> float:
        # Would be better if we could make this less random
        return np.exp(-5 * reward) * 100


class InverseTheilReward(BaseReward):
    """

    """

    def _evaluate(self, g: ig.Graph, *args, **kwargs) -> float:
        metrics_dfs = self.retrieve_dfs(g)

        theil_inequality = {metric: None for metric in self.metrics_names}

        for metric, metric_df in metrics_dfs.items():
            X = metric_df.drop(columns='group').astype(float).to_numpy()
            Y = metric_df.group.to_numpy()
            theil_t = TheilD(X, Y).T[0] if X.sum() > 0 else 0.0
            theil_inequality[metric] = theil_t

        return sum(theil_inequality.values())

    def _reward_scaling(self, reward: float) -> float:
        return np.log(len(self.census_data)) - reward
