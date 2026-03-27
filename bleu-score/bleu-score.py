import numpy as np

def bleu_score(candidate, reference, max_n):
    """
    Compute the BLEU score for a candidate translation.
    """
    clipped_p = []
    for n in range(1, max_n + 1):
        cand_dict = {}
        reff_dict = {}

        for i in range(len(candidate)):
            cand = candidate[i:i+n]
            if len(cand) < n:
                break
            if cand_dict.get(tuple(cand)) is None:
                cand_dict[tuple(cand)] = 1
            else:
                cand_dict[tuple(cand)] += 1
                    
        for i in range(len(reference)):
            reff = reference[i:i+n]
            if len(reff) < n:
                break
            if reff_dict.get(tuple(reff)) is None:
                reff_dict[tuple(reff)] = 1
            else:
                reff_dict[tuple(reff)] += 1
        
        sum_cand = sum(cand_dict.values())
        if sum_cand == 0:
            return 0.0
        
        clipped_p.append(
            sum((
                min(reff_dict.get(key, 0), cand)
                for key, cand in cand_dict.items()
            )) / sum_cand
        )

    BP = np.exp(min(0, 1 - len(reference) / len(candidate)))
    return BP * np.exp(
        np.sum(np.log(clipped_p)) / max_n
    )
