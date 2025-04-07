from .lm_model.LMClass import LMClass
from lm_eval import evaluator
import torch


@torch.no_grad()
def evaluate(lm, args, logger):
    results = {}
    if args.tasks != "":
        t_results = evaluator.simple_evaluate(
            lm,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            limit=None if args.limit == -1 else args.limit,
        )
        results.update(t_results)
        logger.info(results)
        pprint(results)
        # for test of MMLU
        if 'hendrycksTest' in args.tasks:
            all_cors = []
            all_cors_norm = []
            subcat_cors = {subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists}
            cat_cors = {cat: [] for cat in categories}
            cat_cors_norm = {cat: [] for cat in categories}
            for key in t_results['results'].keys():
                if not 'hendrycksTest' in key:
                    continue
                subject = key.split('-')[-1]
                cors = t_results['results'][key]['acc']
                cors_norm = t_results['results'][key]['acc_norm']
                subcats = subcategories[subject]
                for subcat in subcats:
                    subcat_cors[subcat].append(cors)
                    for key in categories.keys():
                        if subcat in categories[key]:
                            cat_cors[key].append(cors)
                            cat_cors_norm[key].append(cors_norm)
                    all_cors.append(cors)
                    all_cors_norm.append(cors_norm)
                    
            for cat in cat_cors:
                cat_acc = np.mean(cat_cors[cat])
                logger.info("Average accuracy {:.4f} - {}".format(cat_acc, cat))
            weighted_acc = np.mean(all_cors)
            logger.info("Average accuracy: {:.4f}".format(weighted_acc))               
    return results

def eval_lm(model, tokenizer, tasks, num_fewshot, batch_size):
    lm = LMClass(model, tokenizer, batch_size)
    lm.seqlen = 2048
    lm.model.eval()
    for param in lm.model.parameters():
        param.requires_grad = False

    results = {}
    t_results = evaluator.simple_evaluate(
        lm,
        model_args=None,
        tasks=tasks.split(","),
        num_fewshot=num_fewshot,
        batch_size=None,
        max_batch_size=None,
        device=None,
        no_cache=True,
        limit=None,
        bootstrap_iters=100000,
        description_dict=None,
        check_integrity=False,
        decontamination_ngrams_path=None,
        write_out=False,
        output_base_path=None,
    )
    results.update(t_results)
    return results