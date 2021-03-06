# -*- coding: utf-8 -*-

"""Test training and HPO mode for RESCAL."""

import pykeen.constants as pkc

from tests.constants import (
    BaseTestTrainingMode,
    set_evaluation_specific_parameters,
    set_hpo_mode_specific_parameters,
    set_training_mode_specific_parameters,
)


class TestTrainingModeForRESCAL(BaseTestTrainingMode):
    """Test that RESCAL can be trained and evaluated correctly in training mode."""

    config = BaseTestTrainingMode.config
    config = set_training_mode_specific_parameters(config=config)
    config[pkc.KG_EMBEDDING_MODEL_NAME] = pkc.RESCAL_NAME
    config[pkc.EMBEDDING_DIM] = 50
    config[pkc.SCORING_FUNCTION_NORM] = 2  # corresponds to L2
    config[pkc.MARGIN_LOSS] = 1  # corresponds to L1

    def test_training(self):
        """Test that RESCAL is trained correctly in training mode."""
        results = self.execute_pipeline(config=self.config)
        self.check_training_mode_without_evaluation(results=results)

    def test_evaluation(self):
        """Test that RESCAL is trained and evaluated correctly in training mode."""
        config = set_evaluation_specific_parameters(config=self.config)
        results = self.execute_pipeline(config=config)
        self.check_training_followed_by_evaluation(results=results)


class TestHPOModeForRESCAL(BaseTestTrainingMode):
    """Test that RESCAL can be trained and evaluated correctly in HPO mode."""

    config = BaseTestTrainingMode.config
    config = set_training_mode_specific_parameters(config=config)
    config[pkc.KG_EMBEDDING_MODEL_NAME] = pkc.RESCAL_NAME
    config[pkc.EMBEDDING_DIM] = [10, 50]
    config[pkc.SCORING_FUNCTION_NORM] = [1, 2]
    config[pkc.MARGIN_LOSS] = [1, 2]

    def test_hpo_mode(self):
        """Test whether HPO mode works correctly for RESCAL."""
        config = set_hpo_mode_specific_parameters(config=self.config)
        results = self.execute_pipeline(config=config)
        self.check_training_followed_by_evaluation(results=results)
