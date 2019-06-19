from typing import Optional

import sys
sys.path.append('./')

import logging
import json
import pickle
import pathlib
import random
from os.path import join
from itertools import groupby
from collections import namedtuple
from operator import attrgetter

import fire
import numpy as np
import tre
from tqdm import tqdm
from sklearn.metrics import auc
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
    

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(FORMATTER)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


Prediction = namedtuple('Prediction', ['score', 'is_correct', 'bag_id', 'bag_labels', 'predicted_label'])
    

def instance_id(mention):
    head = mention['head']['word']
    tail = mention['tail']['word']
    return (head.lower(), tail.lower())


def load_nyt_test_bags(file):
    test_bags = {}
    
    with open(file, 'r') as f:
        nyt_dataset = json.load(f)
    
    for instance, mentions in groupby(sorted(nyt_dataset, key=lambda mention: instance_id(mention)), key=lambda mention: instance_id(mention)):
        head, tail = instance
        mention_instances = []
        
        for mention in mentions:
            mention_instances.append(dict(sentence=mention['sentence'], head=head, tail=tail, relation=mention['relation']))

        test_bags[instance] = mention_instances
    return test_bags


def compute_pr_curve_and_predictions(model_dir: str, test_file: str, archive_filename: str='model.tar.gz',
                                     eval_mode: Optional[str]=None, weights_file: Optional[str]=None,
                                     output_dir: Optional[str]=None, cuda_device: int=0,
                                     predictor_name='tre-classifier', max_instances: Optional[int]=None):
    if eval_mode is not None and eval_mode not in ['one', 'two', 'all']:
        raise ValueError(f"Eval mode '{eval_mode}' not supported.")

    logger.info(f"Loading test file: '{test_file}'")
    test_bags = load_nyt_test_bags(test_file)
    logger.info(f"Test file: '{test_file}' contains {len(test_bags)} bags.")

    archive_path = join(model_dir, archive_filename)
    logger.info(f"Loading model archive: '{archive_path}'")
    if weights_file is not None:
        weights_file = join(model_dir, weights_file)
        logger.info(f"Loading weights file: '{weights_file}'")
    predictor = Predictor.from_archive(load_archive(archive_path,
                                       cuda_device=cuda_device,
                                       weights_file=weights_file),
                                       predictor_name=predictor_name)

    id2label = predictor._model.vocab.get_index_to_token_vocabulary(namespace='labels')
    relation_at_index = list(id2label.values())
    
    num_relation_facts = 0
    
    n_pos = 0

    logger.info(f"Using eval mode '{eval_mode}'.")
    
    predictions = []
    for instance, bag_mentions in tqdm(list(test_bags.items())[:max_instances]):
        if eval_mode is not None:
            if len(bag_mentions) < 2:
                continue

            random.shuffle(bag_mentions)

            if eval_mode == 'one':
                bag_mentions = bag_mentions[:1]
            elif eval_mode == 'two':
                bag_mentions = bag_mentions[:2]

        bag_labels = set([mention['relation'] for mention in bag_mentions])
        bag_labels.discard('NA')
        
        result = predictor.predict_batch_json(bag_mentions)
        
        assert len(result['logits']) == 1
        
        if bag_labels:
            n_pos += 1
        
        num_relation_facts += len(bag_labels)

        # For each bag and positive relation create a prediction
        logits = result['logits'][0]
        for idx, logit in enumerate(logits):
            if relation_at_index[idx] == 'NA':
                continue

            is_correct = relation_at_index[idx] in bag_labels
            predictions.append(Prediction(score=logit,
                                          is_correct=is_correct,
                                          bag_id=instance,
                                          predicted_label=id2label[idx],
                                          bag_labels=bag_labels))

    print(num_relation_facts)
                
    predictions = sorted(predictions, key=attrgetter('score'), reverse=True)
    
    correct = 0
    precision_values = []
    recall_values = []
    for idx, prediction in enumerate(predictions):
        if prediction.is_correct:
            correct += 1
        precision_values.append(correct / (idx+1))
        recall_values.append(correct / num_relation_facts)

    def precision_at(n):
        return (sum([prediction.is_correct for prediction in predictions[:n]]) / n) * 100

    pr_metrics = {
        'P/R AUC': auc(x=recall_values, y=precision_values),
        'Precision@100': precision_at(100),
        'Precision@200': precision_at(200),
        'Precision@300': precision_at(300),
        'Mean': np.mean([precision_at(i) for i in [100, 200, 300]])
    }

    logger.info(f'PR Metrics: {pr_metrics}')

    output_dir = output_dir or model_dir
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    if eval_mode is None:
        with open(join(output_dir, 'pr_metrics.json'), 'w') as pr_metrics_f:
            json.dump(pr_metrics, pr_metrics_f)

        with open(join(output_dir, 'predictions.pkl'), 'wb') as predictions_f:
            pickle.dump(predictions, predictions_f)

        np.save(join(output_dir, 'precision.npy'), precision_values)
        np.save(join(output_dir, 'recall.npy'), recall_values)
    else:
        with open(join(output_dir, f'pr_metrics_{eval_mode}.json'), 'w') as pr_metrics_f:
            json.dump(pr_metrics, pr_metrics_f) 
    

if __name__ == "__main__":
    fire.Fire(compute_pr_curve_and_predictions)
