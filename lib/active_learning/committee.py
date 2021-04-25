import numpy as np
from typing import Callable, Optional, Tuple, List, Any

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

from scipy.stats import entropy
from scipy.spatial import distance

from modAL.models.base import BaseLearner, BaseCommittee
from modAL.utils.validation import check_class_labels, check_class_proba
from modAL.utils.data import modALinput, retrieve_rows
from modAL.utils.selection import multi_argmax, shuffled_argmax
from modAL.uncertainty import uncertainty_sampling
from modAL.disagreement import vote_entropy_sampling, max_std_sampling, KL_max_disagreement, max_disagreement_sampling
from modAL.acquisition import max_EI
from modAL.models import ActiveLearner, Committee


class CustomCommittee(BaseCommittee):
    """
    This class is an abstract model of a committee-based active learning algorithm.

    Args:
        learner_list: A list of ActiveLearners forming the Committee.
        query_strategy: Query strategy function. Committee supports disagreement-based query strategies from
            :mod:`modAL.disagreement`, but uncertainty-based ones from :mod:`modAL.uncertainty` are also supported.
        on_transformed: Whether to transform samples with the pipeline defined by each learner's estimator
            when applying the query strategy.

    Attributes:
        classes_: Class labels known by the Committee.
        n_classes_: Number of classes known by the Committee.

    Examples:

        >>> from sklearn.datasets import load_iris
        >>> from sklearn.neighbors import KNeighborsClassifier
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from modAL.models import ActiveLearner, Committee
        >>>
        >>> iris = load_iris()
        >>>
        >>> # initialize ActiveLearners
        >>> learner_1 = ActiveLearner(
        ...     estimator=RandomForestClassifier(),
        ...     X_training=iris['data'][[0, 50, 100]], y_training=iris['target'][[0, 50, 100]]
        ... )
        >>> learner_2 = ActiveLearner(
        ...     estimator=KNeighborsClassifier(n_neighbors=3),
        ...     X_training=iris['data'][[1, 51, 101]], y_training=iris['target'][[1, 51, 101]]
        ... )
        >>>
        >>> # initialize the Committee
        >>> committee = Committee(
        ...     learner_list=[learner_1, learner_2]
        ... )
        >>>
        >>> # querying for labels
        >>> query_idx, query_sample = committee.query(iris['data'])
        >>>
        >>> # ...obtaining new labels from the Oracle...
        >>>
        >>> # teaching newly labelled examples
        >>> committee.teach(
        ...     X=iris['data'][query_idx].reshape(1, -1),
        ...     y=iris['target'][query_idx].reshape(1, )
        ... )
    """
    def __init__(self, learner_list: List[ActiveLearner], query_strategy: Callable = vote_entropy_sampling,
                 on_transformed: bool = False) -> None:
        super().__init__(learner_list, query_strategy, on_transformed)
        self._set_classes()

    def _set_classes(self):
        """
        Checks the known class labels by each learner, merges the labels and returns a mapping which maps the learner's
        classes to the complete label list.
        """
        # assemble the list of known classes from each learner
        try:
            # if estimators are fitted
            known_classes = tuple(learner.estimator.classes_ for learner in self.learner_list)
        except AttributeError:
            # handle unfitted estimators
            self.classes_ = None
            self.n_classes_ = 0
            return

        self.classes_ = np.unique(
            np.concatenate(known_classes, axis=0),
            axis=0
        )
        self.n_classes_ = len(self.classes_)

    def _add_training_data(self, X: modALinput, y: modALinput):
        super()._add_training_data(X, y)

    def fit(self, X: modALinput, y: modALinput, **fit_kwargs) -> 'BaseCommittee':
        """
        Fits every learner to a subset sampled with replacement from X. Calling this method makes the learner forget the
        data it has seen up until this point and replaces it with X! If you would like to perform bootstrapping on each
        learner using the data it has seen, use the method .rebag()!

        Calling this method makes the learner forget the data it has seen up until this point and replaces it with X!

        Args:
            X: The samples to be fitted on.
            y: The corresponding labels.
            **fit_kwargs: Keyword arguments to be passed to the fit method of the predictor.
        """
        super().fit(X, y, **fit_kwargs)
        self._set_classes()

    def teach(self, X: modALinput, y: modALinput, bootstrap: bool = False, only_new: bool = False, **fit_kwargs) -> None:
        """
        Adds X and y to the known training data for each learner and retrains learners with the augmented dataset.

        Args:
            X: The new samples for which the labels are supplied by the expert.
            y: Labels corresponding to the new instances in X.
            bootstrap: If True, trains each learner on a bootstrapped set. Useful when building the ensemble by bagging.
            only_new: If True, the model is retrained using only X and y, ignoring the previously provided examples.
            **fit_kwargs: Keyword arguments to be passed to the fit method of the predictor.
        """
        super().teach(X, y, bootstrap=bootstrap, only_new=only_new, **fit_kwargs)
        self._set_classes()

    def predict(self, X: modALinput, **predict_proba_kwargs) -> Any:
        """
        Predicts the class of the samples by picking the consensus prediction.

        Args:
            X: The samples to be predicted.
            **predict_proba_kwargs: Keyword arguments to be passed to the :meth:`predict_proba` of the Committee.

        Returns:
            The predicted class labels for X.
        """
        # getting average certainties
        proba = self.predict_proba(X, **predict_proba_kwargs)
        # finding the sample-wise max probability
        max_proba_idx = np.argmax(proba, axis=1)
        # translating label indices to labels
        return self.classes_[max_proba_idx]

    def predict_proba(self, X: modALinput, **predict_proba_kwargs) -> Any:
        """
        Consensus probabilities of the Committee.

        Args:
            X: The samples for which the class probabilities are to be predicted.
            **predict_proba_kwargs: Keyword arguments to be passed to the :meth:`predict_proba` of the Committee.

        Returns:
            Class probabilities for X.
        """
        return np.mean(self.vote_proba(X, **predict_proba_kwargs), axis=1)

    def score(self, X: modALinput, y: modALinput, sample_weight: List[float] = None) -> Any:
        """
        Returns the mean accuracy on the given test data and labels.

        Todo:
            Why accuracy?

        Args:
            X: The samples to score.
            y: Ground truth labels corresponding to X.
            sample_weight: Sample weights.

        Returns:
            Mean accuracy of the classifiers.
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred, sample_weight=sample_weight)

    def vote(self, X: modALinput, **predict_kwargs) -> Any:
        """
        Predicts the labels for the supplied data for each learner in the Committee.

        Args:
            X: The samples to cast votes.
            **predict_kwargs: Keyword arguments to be passed to the :meth:`predict` of the learners.

        Returns:
            The predicted class for each learner in the Committee and each sample in X.
        """
        prediction = np.zeros(shape=(X.shape[0], len(self.learner_list), X.shape[-3],  X.shape[-2], X.shape[-1]))

        for learner_idx, learner in enumerate(self.learner_list):
            prediction[:, learner_idx, :, :, :] = learner.predict(X, **predict_kwargs)

        return prediction

    def vote_proba(self, X: modALinput, **predict_proba_kwargs) -> Any:
        """
        Predicts the probabilities of the classes for each sample and each learner.

        Args:
            X: The samples for which class probabilities are to be calculated.
            **predict_proba_kwargs: Keyword arguments for the :meth:`predict_proba` of the learners.

        Returns:
            Probabilities of each class for each learner and each instance.
        """

        # get dimensions
        n_samples = X.shape[0]
        n_learners = len(self.learner_list)
        # self.n_classes_, X.shape[-3],  X.shape[-2], X.shape[-1]
        proba = np.zeros(shape=(n_samples, n_learners, self.n_classes_, X.shape[-3],  X.shape[-2], X.shape[-1]))
        # print(proba.shape)
        # checking if the learners in the Committee know the same set of class labels
        if check_class_labels(*[learner.estimator for learner in self.learner_list]):
            # known class labels are the same for each learner
            # probability prediction is straightforward

            for learner_idx, learner in enumerate(self.learner_list):
                proba[:, learner_idx, :, :, :, :] = learner.predict_proba(X, **predict_proba_kwargs)

        else:
            for learner_idx, learner in enumerate(self.learner_list):
                proba[:, learner_idx, :, :, :, :] = check_class_proba(
                    proba=learner.predict_proba(X, **predict_proba_kwargs),
                    known_labels=learner.estimator.classes_,
                    all_labels=self.classes_
                )

        return proba


def JS_divergence(probabilities):

  num_models = len(probabilities)

  part1 = 0
  part2 = 0

  #### if probabilities is list of list where nested list is list of each model's probability

  sum1 = [sum(x) for x in zip(*probabilities)]
  # print(len(probabilities[0][1]))
  new_sum = [x / num_models for x in sum1]

  part1 = entropy(new_sum,base=len(probabilities[0]))

  sum2 = 0
  for prob_list in probabilities:
    sum2 = sum2 + entropy(prob_list,base=len(probabilities[0]))

  part2 = sum2/num_models

  JS_div = part1-part2

  return JS_div


def JSD_max_disagreement(committee, X) -> np.ndarray:
  """
  Args:
  committee: The :class:`modAL.models.BaseCommittee` instance for which the max disagreement is to be calculated.
  X: The data for which the max disagreement is to be calculated.
  **predict_proba_kwargs: Keyword arguments for the :meth:`predict_proba` of the Committee.

  Returns:
  Max disagreement of the Committee for the samples in X.
  """
  try:
    p_vote = committee.vote_proba(X)
    # print(p_vote.shape)
  except NotFittedError:
    return np.zeros(shape=(X.shape[0],))
  p_vote = np.reshape(p_vote, (p_vote.shape[0], p_vote.shape[1], p_vote.shape[2] * p_vote.shape[-1]**3))
  # print(p_vote.shape)

  JSE_Xtest = []

  for item in p_vote:
    JSE_Xtest.append(JS_divergence(item))

  return np.array(JSE_Xtest)


def JSD_max_disagreement_sampling(committee, X,
                              n_instances, random_tie_break=False,
                             ):
    """
    Maximum disagreement sampling strategy.

    Args:
        committee: The committee for which the labels are to be queried.
        X: The pool of samples to query from.
        n_instances: Number of samples to be queried.
        random_tie_break: If True, shuffles utility scores to randomize the order. This
            can be used to break the tie when the highest utility score is not unique.
        **disagreement_measure_kwargs: Keyword arguments to be passed for the disagreement
         measure function.

    Returns:
        The indices of the instances from X chosen to be labelled;
        the instances from X chosen to be labelled.
    """
    disagreement = JSD_max_disagreement(committee, X)

    if not random_tie_break:
        query_idx = multi_argmax(disagreement, n_instances=n_instances)
    else:
        query_idx = shuffled_argmax(disagreement, n_instances=n_instances)

    return query_idx, X[query_idx]


def KL_max_disagreement_custom(committee: BaseCommittee, X: modALinput, **predict_proba_kwargs) -> np.ndarray:
    """
    Calculates the max disagreement for the Committee. First it computes the class probabilties of X for each learner in
    the Committee, then calculates the consensus probability distribution by averaging the individual class
    probabilities for each learner. Then each learner's class probabilities are compared to the consensus distribution
    in the sense of Kullback-Leibler divergence. The max disagreement for a given sample is the argmax of the KL
    divergences of the learners from the consensus probability.

    Args:
        committee: The :class:`modAL.models.BaseCommittee` instance for which the max disagreement is to be calculated.
        X: The data for which the max disagreement is to be calculated.
        **predict_proba_kwargs: Keyword arguments for the :meth:`predict_proba` of the Committee.

    Returns:
        Max disagreement of the Committee for the samples in X.
    """
    try:
        p_vote = committee.vote_proba(X, **predict_proba_kwargs)
        p_vote = np.reshape(p_vote, (p_vote.shape[0], p_vote.shape[1], p_vote.shape[2] * p_vote.shape[-1]**3))
    except NotFittedError:
        return np.zeros(shape=(X.shape[0],))

    p_consensus = np.mean(p_vote, axis=1)

    learner_KL_div = np.zeros(shape=(X.shape[0], len(committee)))
    for learner_idx, _ in enumerate(committee):
        learner_KL_div[:, learner_idx] = entropy(np.transpose(p_vote[:, learner_idx, :]), qk=np.transpose(p_consensus))

    return np.max(learner_KL_div, axis=1)


def KL_max_disagreement_sampling_custom(committee: BaseCommittee, X: modALinput,
                              n_instances: int = 1, random_tie_break=False,
                              **disagreement_measure_kwargs) -> Tuple[np.ndarray, modALinput]:
    """
    Maximum disagreement sampling strategy.

    Args:
        committee: The committee for which the labels are to be queried.
        X: The pool of samples to query from.
        n_instances: Number of samples to be queried.
        random_tie_break: If True, shuffles utility scores to randomize the order. This
            can be used to break the tie when the highest utility score is not unique.
        **disagreement_measure_kwargs: Keyword arguments to be passed for the disagreement
         measure function.

    Returns:
        The indices of the instances from X chosen to be labelled;
        the instances from X chosen to be labelled.
    """
    disagreement = KL_max_disagreement_custom(committee, X, **disagreement_measure_kwargs)
    # print(disagreement.shape, "disagreement")

    if not random_tie_break:
        query_idx = multi_argmax(disagreement, n_instances=n_instances)
        # print(query_idx.shape, "query_idx")
    else:
        query_idx = shuffled_argmax(disagreement, n_instances=n_instances)

    return query_idx, X[query_idx]


def consensus_entropy_custom(committee: BaseCommittee, X: modALinput, **predict_proba_kwargs) -> np.ndarray:
    """
    Calculates the consensus entropy for the Committee. First it computes the class probabilties of X for each learner
    in the Committee, then calculates the consensus probability distribution by averaging the individual class
    probabilities for each learner. The entropy of the consensus probability distribution is the vote entropy of the
    Committee, which is returned.

    Args:
        committee: The :class:`modAL.models.BaseCommittee` instance for which the consensus entropy is to be calculated.
        X: The data for which the consensus entropy is to be calculated.
        **predict_proba_kwargs: Keyword arguments for the :meth:`predict_proba` of the Committee.

    Returns:
        Consensus entropy of the Committee for the samples in X.
    """
    try:
        proba = committee.predict_proba(X, **predict_proba_kwargs)
        proba = np.reshape(proba, (proba.shape[0], proba.shape[1]*proba.shape[-1]**3))
    except NotFittedError:
        return np.zeros(shape=(X.shape[0],))

    entr = np.transpose(entropy(np.transpose(proba)))
    
    return entr


def consensus_entropy_sampling_custom(committee: BaseCommittee, X: modALinput,
                               n_instances: int = 1, random_tie_break=False,
                               **disagreement_measure_kwargs) -> Tuple[np.ndarray, modALinput]:
    """
    Consensus entropy sampling strategy.

    Args:
        committee: The committee for which the labels are to be queried.
        X: The pool of samples to query from.
        n_instances: Number of samples to be queried.
        random_tie_break: If True, shuffles utility scores to randomize the order. This
            can be used to break the tie when the highest utility score is not unique.
        **disagreement_measure_kwargs: Keyword arguments to be passed for the disagreement
            measure function.

    Returns:
        The indices of the instances from X chosen to be labelled;
        the instances from X chosen to be labelled.
    """
    disagreement = consensus_entropy_custom(committee, X, **disagreement_measure_kwargs)

    if not random_tie_break:
        query_idx = multi_argmax(disagreement, n_instances=n_instances)
    else:
        query_idx = shuffled_argmax(disagreement, n_instances=n_instances)

    return query_idx, X[query_idx]