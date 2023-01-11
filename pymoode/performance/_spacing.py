import numpy as np
from scipy.spatial.distance import pdist, squareform
from pymoo.core.indicator import Indicator
from pymoo.indicators.distance_indicator import at_least_2d_array, derive_ideal_and_nadir_from_pf


# =========================================================================================================
# Implementation
# =========================================================================================================

class SpacingIndicator(Indicator):

    def __init__(self,
                 metric="cityblock",
                 pf=None,
                 zero_to_one=False,
                 ideal=None,
                 nadir=None):
        """Spacing indicator
        The smaller the value this indicator assumes, the most uniform is the distribution of elements on the pareto front.
        
        Parameters
        ----------
        metric : str, optional
            Distance metric parsed to scipy.spatial.distance.pdist, by default "cityblock"
        
        pf : 2d array, optional
            Pareto front, by default None
        
        zero_to_one : bool, optional
            Whether or not the objective values should be normalized in calculations, by default False
        
        ideal : 1d array, optional
            Ideal point, by default None
        
        nadir : 1d array, optional
            Nadir point, by default None
        """

        # the pareto front if necessary to calculate the indicator
        pf = at_least_2d_array(pf, extend_as="row")
        ideal, nadir = derive_ideal_and_nadir_from_pf(pf, ideal=ideal, nadir=nadir)

        super().__init__(pf=pf,
                         zero_to_one=zero_to_one,
                         ideal=ideal,
                         nadir=nadir)
        
        self.metric = metric
    
    def do(self, F, *args, **kwargs):
        """Obtain the spacing indicator given a Pareto front

        Parameters
        ----------
        F : numpy.array (n_samples, n_obj)
            Pareto front

        Returns
        -------
        float
            Spacing indicator
        """
        return super().do(F, *args, **kwargs)

    def _do(self, F, *args, **kwargs):

        # Get F dimensions
        n_points, n_obj = F.shape

        # knn
        D = squareform(pdist(F, metric=self.metric))
        d = np.partition(D, 1, axis=1)[:, 1]
        dm = np.mean(d)

        # Get spacing
        S = np.sqrt(np.sum(np.square(d - dm)) / n_points)

        return S
