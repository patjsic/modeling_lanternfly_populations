import pandas as pd
import pymc as pm
from typing import Dict
from pymc_extras.model_builder import ModelBuilder

class GMRF(ModelBuilder):
    #Name of model
    _model_type = "GMRF"
    version = "0.1"

    def build_model(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        X_values = X["observation_count"].fillna(0).values

        with pm.Model() as self.model:
            #Create mutable data containers
            # population_counts = 
        
    @staticmethod
    def get_default_model_config() -> Dict:
        model_config: Dict = {
            "lambda_alpha": 1.0,
            "lambda_beta": 1.0,
            "sigma_scale": 1.0
        }
        return model_config
    
    @staticmethod
    def get_default_sampler_config() -> Dict:
        sampler_config: Dict = {
            "draws": 200,
            "tune": 200,
            "chains": 3,
            "target_accept": 0.95
        }
        return sampler_config
    
    def _save_input_params(self, idata) -> None:
        """
        Saves any additional model parameters (other than the dataset) to the idata object.

        These parameters are stored within `idata.attrs` using keys that correspond to the parameter names.
        If you don't need to store any extra parameters, you can leave this method unimplemented.

        Example:
            For saving customer IDs provided as an 'customer_ids' input to the model:
            self.customer_ids = customer_ids.values #this line is done outside of the function, preferably at the initialization of the model object.
            idata.attrs["customer_ids"] = json.dumps(self.customer_ids.tolist())  # Convert numpy array to a JSON-serializable list.
        """
        pass

        pass