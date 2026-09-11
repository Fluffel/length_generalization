import random
import string

from abc import ABC, abstractmethod
from copy import deepcopy
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


class customTokenizer():
    def __init__(self, vocab: list[str]):
        normal_tkn_num = len(vocab) # each element is a token

        self.bos_token = "<bos>"
        self.sep_token = "<sep>"
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.bos_token_id = normal_tkn_num
        self.sep_token_id = normal_tkn_num + 1
        self.eos_token_id = normal_tkn_num + 2
        self.pad_token_id = normal_tkn_num + 3
        self.special_token_ids = [self.bos_token_id, self.sep_token_id, self.eos_token_id, self.pad_token_id]
        self.special_tokens = [self.bos_token, self.sep_token, self.eos_token, self.pad_token]
        assert all(t not in vocab for t in self.special_tokens)
        
        # self.vocab = {"0": 0, "1": 1}
        self.vocab = {t: i for i, t in enumerate(vocab)}
        self.vocab[self.bos_token] = self.bos_token_id
        self.vocab[self.sep_token] = self.sep_token_id
        self.vocab[self.eos_token] = self.eos_token_id
        self.vocab[self.pad_token] = self.pad_token_id

        self.vocab_inv = {v: k for k, v in self.vocab.items()}
        self.padding_side = "right"

    def __call__(self, strings: list[str] | str, **kwargs):
        # this func is not used, since the data generator does not generate str
        # string is tokenized by white space
        if type(strings) == str:
            strings = [strings]
        ids = []
        strings = [s.split(" ") for s in strings]
        max_len = max(map(lambda x: len(x), strings))
        for s in strings:
            ids.append( list(map(lambda x: self.vocab[x], s)) + [self.pad_token_id] * (max_len-len(s)) )

        return {"input_ids": torch.LongTensor(ids)}

    def convert_ids_to_tokens(self, ids: list[int], rm_special=False):
        if rm_special:
            return [self.vocab_inv[i] for i in ids if i not in self.special_token_ids]
        else:
            return list(map(lambda x: self.vocab_inv[x], ids))

    def __len__(self):
        return len(self.vocab)
    
class CustomDataset(IterableDataset):
    def __init__(self, n_positions: int, add_positional_offset: bool):
        super().__init__()
        self._n_positions = n_positions
        self._add_positional_offset = add_positional_offset

    @property
    def n_positions(self):
        return self._n_positions
    
    def get_pos_ids(self, instance_length, max_offset):
        """
        Get the positional ids of length instance_length. If add_positional_offset is set to True, a random offset between [0...max_offset] is added to those ids.
        """
        offset = 0
        if self._add_positional_offset:
            offset = random.randint(0, max(0, max_offset))
        return list(range(offset, instance_length + offset))

    
    
class BinaryMajorityDataset(CustomDataset):
    def __init__(self, length_range: tuple[int, int], max_test_length: int, add_positional_offset: bool = True):
        super().__init__(max_test_length + 4, add_positional_offset) # bos, sep, ans, eos

        self.tokenizer = customTokenizer(["0", "1"])
        assert len(self.tokenizer) == 6
        self.range_min, self.range_max = length_range
        self.range_min = max(1, self.range_min)
        self.max_test_length = max_test_length
        assert (max_test_length >= self.range_max) or (max_test_length == -1)   # the pos emb is initialized based on max_test_length

    def __iter__(self):
        while True:
            length = random.randint(self.range_min, self.range_max)
            while True:
                num_zero = random.randint(0, length)
                if num_zero != length-num_zero:
                    break
            instance = [0, ] * num_zero + [1, ] * (length - num_zero)
            random.shuffle(instance)
            ans = 0 if num_zero > length-num_zero else 1

            instance.insert(0, self.tokenizer.bos_token_id)
            instance.append(self.tokenizer.sep_token_id)
            instance.append(ans)
            instance.append(self.tokenizer.eos_token_id)

            label = deepcopy(instance)
            # setting some tokens to [pad] will make the loss on these tokens (as pred targets) be ignored
            label[:length+2] = [self.tokenizer.pad_token_id,] * (length+2)   # bos + bits.. + sep 
            
            # if self.max_test_length != -1:
            #     offset = random.randint(0, self.max_test_length - length)
            # else:
            #     offset = 0
            # pos_ids = list(range(offset, len(instance)+offset))
            pos_ids = self.get_pos_ids(len(instance), self.max_test_length - length)

            yield instance, pos_ids, label


class MajorityDataset(CustomDataset):
    def __init__(self, length_range: tuple[int, int], max_test_length: int, add_positional_offset: bool = True):
        super().__init__(max_test_length + 4, add_positional_offset)      # bos, sep, ans, eos

        self.tokenizer = customTokenizer(list(string.ascii_lowercase))
        self.range_min, self.range_max = length_range
        self.range_min = max(1, self.range_min)
        self.max_test_length = max_test_length
        assert (max_test_length >= self.range_max) or (max_test_length == -1)   # the pos emb is initialized based on max_test_length

    def __iter__(self):
        while True:
            length = random.randint(self.range_min, self.range_max)
            while True:
                instance = random.choices(range(len(self.tokenizer)-4), k=length)
                most_common = Counter(instance).most_common(2)
                if len(most_common) < 2 or most_common[0][1] > most_common[1][1]:
                    break
            ans = most_common[0][0]

            instance.insert(0, self.tokenizer.bos_token_id)
            instance.append(self.tokenizer.sep_token_id)
            instance.append(ans)
            instance.append(self.tokenizer.eos_token_id)

            label = deepcopy(instance)
            # setting some tokens to [pad] will make the loss on these tokens (as pred targets) be ignored
            label[:length+2] = [self.tokenizer.pad_token_id,] * (length+2)   # bos + bits.. + sep 
            
            # if self.max_test_length != -1:
            #     offset = random.randint(0, self.max_test_length - length)
            # else:
            #     offset = 0
            # pos_ids = list(range(offset, len(instance)+offset))
            pos_ids = self.get_pos_ids(len(instance), self.max_test_length - length)

            yield instance, pos_ids, label

class BinaryMajorityInterleaveDataset(CustomDataset):
    def __init__(self, length_range: tuple[int, int], max_test_length: int, period: int, add_positional_offset: bool = True):
        super().__init__(max_test_length + 6, add_positional_offset)       # ans

        self.tokenizer = customTokenizer(["0", "1"])
        assert len(self.tokenizer) == 6
        self.range_min, self.range_max = length_range
        self.range_min = max(3, self.range_min)
        self.max_test_length = max_test_length
        self.period = period
        assert (max_test_length >= self.range_max) or (max_test_length == -1)    # the pos emb is initialized based on max_test_length

    def __iter__(self):
        while True:
            total_length = random.randint(self.range_min, self.range_max)
            length = round(total_length / self.period)
            if length * self.period > self.range_max:
                length -= 1
            if length * self.period < self.range_min:
                length += 1
            
            instances = []
            answers = []
            for i in range(self.period):
                while True:
                    num_zero = random.randint(0, length)
                    if num_zero != length-num_zero:
                        break
                instance = [0, ] * num_zero + [1, ] * (length - num_zero)
                random.shuffle(instance)
                instances.append(instance)

                ans = 0 if num_zero > length-num_zero else 1
                answers.append(ans)

            whole_instance = [val for tup in zip(*instances) for val in tup]

            whole_instance.insert(0, self.tokenizer.bos_token_id)
            whole_instance.append(self.tokenizer.sep_token_id)
            whole_instance.extend(answers)
            whole_instance.append(self.tokenizer.eos_token_id)

            label = deepcopy(whole_instance)
            # setting some tokens to [pad] will make the loss on these tokens (as pred targets) be ignored
            label[:length*self.period+2] = [self.tokenizer.pad_token_id,] * (length*self.period+2)   # bos + bits.. + sep 
            
            # if self.max_test_length != -1:
            #     offset = random.randint(0, self.max_test_length - length*self.period)
            # else:
            #     offset = 0
            # pos_ids = list(range(offset, len(whole_instance)+offset))
            pos_ids = self.get_pos_ids(len(whole_instance), self.max_test_length - length * self.period)

            yield whole_instance, pos_ids, label


class UniqueCopyDataset(CustomDataset):
    def __init__(self, length_range: tuple[int, int], max_test_length: int, add_positional_offset: bool = True):
        super().__init__(max_test_length*2 + 3, add_positional_offset)    # bos, sep, eos

        self.tokenizer = customTokenizer([str(i) for i in range(max_test_length)]) 
        self.range_min, self.range_max = length_range
        self.range_min = max(1, self.range_min)
        self.max_test_length = max_test_length
        assert len(self.tokenizer) - 4 >= max_test_length
        assert (max_test_length >= self.range_max) or (max_test_length == -1)    # the pos emb is initialized based on max_test_length

    def __iter__(self):
        while True:
            length = random.randint(self.range_min, self.range_max)     # length of string to be copied
            
            temp = random.sample(range(len(self.tokenizer)-4), length)
            instance = [self.tokenizer.bos_token_id]
            instance.extend(temp)
            instance.append(self.tokenizer.sep_token_id)
            instance.extend(temp)
            instance.append(self.tokenizer.eos_token_id)

            label = deepcopy(instance)
            # setting some tokens to [pad] will make the loss on these tokens (as pred targets) be ignored
            label[:length+2] = [self.tokenizer.pad_token_id,] * (length+2)   # bos + ... + sep 
            
            # if self.max_test_length != -1:
            #     offset = random.randint(0, (self.max_test_length - length) * 2)
            # else:
            #     offset = 0
            # pos_ids = list(range(offset, len(instance)+offset))
            pos_ids = self.get_pos_ids(len(instance), (self.max_test_length - length) * 2)

            yield instance, pos_ids, label


class RepeatCopyDataset(CustomDataset):
    def __init__(self, length_range: tuple[int, int], max_test_length: int, add_positional_offset: bool = True):
        super().__init__(max_test_length*2 + 3, add_positional_offset)  # bos, sep, eos

        self.tokenizer = customTokenizer(["a", "b"])
        self.range_min, self.range_max = length_range
        self.range_min = max(1, self.range_min)
        self.max_test_length = max_test_length
        assert (max_test_length >= self.range_max) or (max_test_length == -1)    # the pos emb is initialized based on max_test_length

    def __iter__(self):
        while True:
            length = random.randint(self.range_min, self.range_max)     # length of string to be copied
            
            temp = random.choices(range(len(self.tokenizer)-4), k=length)
            instance = [self.tokenizer.bos_token_id]
            instance.extend(temp)
            instance.append(self.tokenizer.sep_token_id)
            instance.extend(temp)
            instance.append(self.tokenizer.eos_token_id)

            label = deepcopy(instance)
            # setting some tokens to [pad] will make the loss on these tokens (as pred targets) be ignored
            label[:length+2] = [self.tokenizer.pad_token_id,] * (length+2)   # bos + ... + sep 
            
            # if self.max_test_length != -1:
            #     offset = random.randint(0, (self.max_test_length - length) * 2)
            # else:
            #     offset = 0
            # pos_ids = list(range(offset, len(instance)+offset))
            pos_ids = self.get_pos_ids(len(instance), (self.max_test_length - length) * 2)

            yield instance, pos_ids, label


class SortDataset(CustomDataset):
    def __init__(self, length_range: tuple[int, int], max_test_length: int, add_positional_offset: bool = True):
        super().__init__(max_test_length*2 + 3, add_positional_offset) # bos, sep, eos

        self.tokenizer = customTokenizer([str(i) for i in range(max_test_length)])
        self.range_min, self.range_max = length_range
        self.range_min = max(1, self.range_min)
        self.max_test_length = max_test_length
        assert len(self.tokenizer) - 4 >= max_test_length
        assert (max_test_length >= self.range_max) or (max_test_length == -1)    # the pos emb is initialized based on max_test_length

    def __iter__(self):
        while True:
            length = random.randint(self.range_min, self.range_max)     # length of string to be copied

            temp = random.sample(range(len(self.tokenizer)-4), length)
            instance = [self.tokenizer.bos_token_id]
            instance.extend(temp)
            instance.append(self.tokenizer.sep_token_id)
            instance.extend(sorted(temp))
            instance.append(self.tokenizer.eos_token_id)

            label = deepcopy(instance)
            # setting some tokens to [pad] will make the loss on these tokens (as pred targets) be ignored
            label[:length+2] = [self.tokenizer.pad_token_id,] * (length+2)   # bos + bits.. + sep 
            
            # if self.max_test_length != -1:
            #     offset = random.randint(0, (self.max_test_length - length) * 2)
            # else:
            #     offset = 0
            # pos_ids = list(range(offset, len(instance)+offset))
            pos_ids = self.get_pos_ids(len(instance), (self.max_test_length - length) * 2)

            yield instance, pos_ids, label


class ParityDataset(CustomDataset):
    def __init__(self, length_range: tuple[int, int], max_test_length: int, add_positional_offset: bool = True):
        super().__init__(max_test_length + 4, add_positional_offset)  # bos, sep, ans, eos

        self.tokenizer = customTokenizer(["0", "1", "e", "o"])       # even, odd
        self.range_min, self.range_max = length_range
        self.max_test_length = max_test_length
        assert (max_test_length >= self.range_max) or (max_test_length == -1)    # the pos emb is initialized based on max_test_length

    def __iter__(self):
        while True:
            length = random.randint(self.range_min, self.range_max) 
            num_ones = random.randint(0, length)
            temp = [self.tokenizer.vocab["1"]] * num_ones + [self.tokenizer.vocab["0"]] * (length - num_ones)
            random.shuffle(temp)
            ans = self.tokenizer.vocab["e"] if num_ones % 2 == 0 else self.tokenizer.vocab["o"]

            instance = [self.tokenizer.bos_token_id]
            instance.extend(temp)
            instance.append(self.tokenizer.sep_token_id)
            instance.append(ans)
            instance.append(self.tokenizer.eos_token_id)

            label = deepcopy(instance)
            # setting some tokens to [pad] will make the loss on these tokens (as pred targets) be ignored
            label[:length+2] = [self.tokenizer.pad_token_id,] * (length+2)   # bos + bits.. + sep 
            
            # if self.max_test_length != -1:
            #     offset = random.randint(0, self.max_test_length - length)
            # else:
            #     offset = 0
            # pos_ids = list(range(offset, len(instance)+offset))
            pos_ids = self.get_pos_ids(len(instance), self.max_test_length - length)

            yield instance, pos_ids, label

class AdditionDataset(CustomDataset):
    def __init__(self, length_range: tuple[int, int], max_test_length: int, add_positional_offset: bool = True):
        super().__init__(max_test_length*2, add_positional_offset)  # bos, ans, eos

        self.tokenizer = customTokenizer(["0", "1", "+", "="])
        self.range_min, self.range_max = length_range
        self.range_min = max(4, self.range_min)
        self.max_test_length = max_test_length
        assert (max_test_length >= self.range_max) or (max_test_length == -1)    # the pos emb is initialized based on max_test_length

    def __iter__(self):
        while True:
            length = random.randint(self.range_min, self.range_max)     # length of string to be copied

            len_operand1 = random.randint(1, length-3)
            len_operand2 = length - 2 - len_operand1
            
            if len_operand1 > 1:
                operand1 = ["1"] + random.choices(["0", "1"], k=len_operand1-1)
            else:
                operand1 = random.choices(["0", "1"], k=1)
            if len_operand2 > 1:
                operand2 = ["1"] + random.choices(["0", "1"], k=len_operand2-1)
            else:
                operand2 = random.choices(["0", "1"], k=1)

            ans = int("0b" + "".join(operand1), 2) + int("0b" + "".join(operand2), 2)
            ans = list(bin(ans)[2:])

            instance = [self.tokenizer.bos_token]
            instance.extend(operand1)
            instance.append("+")
            instance.extend(operand2)
            instance.append("=")
            instance.extend(ans)
            instance.append(self.tokenizer.eos_token)

            instance = list(map(lambda x: self.tokenizer.vocab[x], instance))

            label = deepcopy(instance)
            # setting some tokens to [pad] will make the loss on these tokens (as pred targets) be ignored
            label[:length+1] = [self.tokenizer.pad_token_id,] * (length+1)   # bos + bits.. + sep 
            
            # if self.max_test_length != -1:
            #     offset = random.randint(0, self.max_test_length*2 - len(instance))
            # else:
            #     offset = 0
            # pos_ids = list(range(offset, len(instance)+offset))
            pos_ids = self.get_pos_ids(len(instance), self.max_test_length * 2 - len(instance))

            yield instance, pos_ids, label

# ── Monoid presets ──────────────────────────────────────────────────────────
# Each returns (op, identity, monoid_size) where op: (int, int) -> int
# operates on monoid element indices 0..monoid_size-1.

def parity_monoid():
    """Z_2 under XOR. Elements: {0, 1}."""
    return (lambda a, b: a ^ b), 0, 2

def cyclic_monoid(n: int):
    """Z_n under addition mod n. Elements: {0, 1, ..., n-1}."""
    return (lambda a, b: (a + b) % n), 0, n

def monoid_from_cayley_table(table: list[list[int]], identity: int):
    """
    Arbitrary finite monoid from a Cayley (multiplication) table.
    table[i][j] = op(i, j). identity is the index of the identity element.
    """
    monoid_size = len(table)
    return (lambda a, b: table[a][b]), identity, monoid_size


class MQARWordProblemDataset(CustomDataset):
    def __init__(self, length_range: tuple[int, int],
                 max_test_length: int, add_positional_offset: bool = True, key_size: int = 32,
                 query_fraction_upper: float = 0.2, query_fraction_lower: float = 0.2, monoid_type: str = "parity", monoid_n: int = 2):
        """
        MQAR Word Problem dataset.

        Args:
            tokenizer: customTokenizer whose vocab is [k0..k_{K-1}, m0..m_{M-1}] + specials.
            length_range: (min, max) for the content length (2T + Q).
            max_test_length: max content length for position ID offset (-1 for eval).
            key_size: |K|, number of distinct keys.
            monoid_size: |M|, number of monoid elements.
            query_fraction_upper: upper bound for the fraction of content length devoted to queries.
            query_fraction_lower: lower bound for the fraction of content length devoted to queries.
            op: binary monoid operation on element indices (0..M-1) -> (0..M-1).
            identity: index of the monoid identity element.
        """
        super().__init__(-1, add_positional_offset) # placeholder

        match monoid_type:
            case "parity":
                self.op, self.identity, self.monoid_size = parity_monoid()
            case "cyclic":
                self.op, self.identity, self.monoid_size = cyclic_monoid(monoid_n)

        vocab = [f"k{i}" for i in range(key_size)] + [f"m{i}" for i in range(self.monoid_size)]
        self.tokenizer = customTokenizer(vocab)
        self.range_min, self.range_max = length_range
        self.max_test_length = max_test_length
        self.key_size = key_size
        self.query_fraction_upper = query_fraction_upper
        self.query_fraction_lower = query_fraction_lower


        # Key token IDs: 0..key_size-1
        # Monoid token IDs: key_size..key_size+monoid_size-1
        self.monoid_token_offset = key_size

        # Validate constraints
        assert 0 < query_fraction_lower <= query_fraction_upper < 1, "query_fraction must be in (0, 1)"
        # Ensure we can always form valid instances at range_min
        self._validate_length(self.range_min)

    def _derive_T_Q(self, length: int):
        """Derive T (num update pairs) and Q (num queries) from content length."""
        query_fraction = random.uniform(self.query_fraction_lower, self.query_fraction_upper)
   
        Q = max(1, round(query_fraction * length))
        T = (length - Q) // 2
        # Clamp: need T >= Q (enough keys to query) and T >= 1
        if T < Q:
            T = Q
        T = min(T, self.key_size)  # can't exceed available keys
        Q = min(Q, T)  # can't query more keys than we have
        return T, Q

    def _validate_length(self, length: int):
        T, Q = self._derive_T_Q(length)
        assert T >= 1, f"Cannot form valid instance: T={T} at length={length}"
        assert Q >= 1, f"Cannot form valid instance: Q={Q} at length={length}"
        assert T <= self.key_size, (
            f"key_size={self.key_size} too small for T={T} at length={length}")

    @property
    def n_positions(self):
        """Compute max sequence length (for n_positions in GPT2Config)."""
        T, Q = self._derive_T_Q(self.max_test_length)
        # <bos> k m k m ... <sep> q q ... <sep> answer <eos>
        return 2 * T + Q + 5  # bos + 2T + sep + Q + sep + answer + eos

    def __iter__(self):
        while True:
            length = random.randint(self.range_min, self.range_max)
            T, Q = self._derive_T_Q(length)

            # Sample T unique keys (as token IDs 0..key_size-1)
            keys = random.sample(range(self.key_size), T)
            # Sample T monoid elements (as monoid indices 0..monoid_size-1)
            values = [random.randint(0, self.monoid_size - 1) for _ in range(T)]
            kv_map = dict(zip(keys, values))

            # Sample Q query keys without replacement from the T keys
            query_keys = random.sample(keys, Q)

            # Compute answer: left-fold retrieved values with monoid op
            answer_idx = self.identity
            for qk in query_keys:
                answer_idx = self.op(answer_idx, kv_map[qk])

            # Build token sequence:
            # <bos> k1 m1 k2 m2 ... kT mT <sep> q1 ... qQ <sep> answer <eos>
            instance = [self.tokenizer.bos_token_id]
            for k, v in zip(keys, values):
                instance.append(k)                              # key token ID
                instance.append(self.monoid_token_offset + v)   # monoid token ID
            instance.append(self.tokenizer.sep_token_id)
            for qk in query_keys:
                instance.append(qk)                             # key token ID
            instance.append(self.tokenizer.sep_token_id)
            instance.append(self.monoid_token_offset + answer_idx)  # answer token ID
            instance.append(self.tokenizer.eos_token_id)

            # Label: mask everything before the answer
            label = deepcopy(instance)
            mask_len = 2 * T + Q + 3  # bos + 2T pairs + sep + Q queries + sep
            label[:mask_len] = [self.tokenizer.pad_token_id] * mask_len

            # Position IDs with random offset
            # if self.max_test_length != -1:
            #     n_pos = self.n_positions()
            #     offset = random.randint(0, max(0, n_pos - len(instance)))
            # else:
            #     offset = 0
            # pos_ids = list(range(offset, len(instance) + offset))
            pos_ids = self.get_pos_ids(len(instance), max(0, self.n_positions - len(instance)))

            yield instance, pos_ids, label


class FlipFlopDataset(CustomDataset):
    """
    Flip-flop language modeling (Liu et al., 2023): simulates a 1-bit register.

    Each instruction token (w, i, r) is followed by a bit (0 or 1):
      - w <bit>: write <bit> to the register
      - i <bit>: ignore (bit is random noise, register unchanged)
      - r <bit>: read (bit = current register value)

    The first instruction is always w (register init). Reads interleaved
    in the input carry the correct register bit so the model sees the full
    autoregressive history. The last instruction is r, whose bit is the
    single-token answer after <sep>.

    Sequence: <bos> w 1 i 0 w 0 r 0 i 1 w 1 r <sep> 1 <eos>

    length_range controls the number of instructions (not tokens).
    Following FFL(p), ignore_fraction is the fraction of middle
    instructions that are ignores; the rest are split equally between
    reads and writes (default p=0.8 -> 10% read, 10% write).
    """

    def __init__(self, length_range: tuple[int, int], max_test_length: int,
                 ignore_fraction: float = 0.8, add_positional_offset: bool = True):
        super().__init__(max_test_length * 2 + 3, add_positional_offset)

        self.tokenizer = customTokenizer(["w", "i", "r", "0", "1"])
        self.range_min, self.range_max = length_range
        self.range_min = max(2, self.range_min)
        self.range_max = max(2, self.range_max)
        self.max_test_length = max_test_length
        self.ignore_fraction = ignore_fraction
        assert 0.0 <= ignore_fraction < 1.0
        assert (max_test_length >= self.range_max) or (max_test_length == -1)

    def __iter__(self):
        w = self.tokenizer.vocab["w"]
        i_tok = self.tokenizer.vocab["i"]
        r = self.tokenizer.vocab["r"]
        bit_0 = self.tokenizer.vocab["0"]
        bit_1 = self.tokenizer.vocab["1"]

        write_fraction = (1.0 - self.ignore_fraction) / 2

        while True:
            num_instr = random.randint(self.range_min, self.range_max)

            # (instruction_token, bit_value) pairs for all but the last read
            pairs: list[tuple[int, int]] = []
            register = -1

            # First instruction: always write
            bit = random.randint(0, 1)
            pairs.append((w, bit))
            register = bit

            # Middle instructions
            for _ in range(1, num_instr - 1):
                roll = random.random()
                if roll < self.ignore_fraction:
                    bit = random.randint(0, 1)
                    pairs.append((i_tok, bit))
                elif roll < self.ignore_fraction + write_fraction:
                    bit = random.randint(0, 1)
                    pairs.append((w, bit))
                    register = bit
                else:
                    pairs.append((r, register))

            # Build token sequence
            instance = [self.tokenizer.bos_token_id]
            for instr, b in pairs:
                instance.append(instr)
                instance.append(bit_0 if b == 0 else bit_1)
            # Last instruction: r without its bit (that is the answer)
            instance.append(r)
            instance.append(self.tokenizer.sep_token_id)
            instance.append(bit_0 if register == 0 else bit_1)
            instance.append(self.tokenizer.eos_token_id)

            label = deepcopy(instance)
            # Mask everything up to and including <sep>
            # bos + 2*(num_instr-1) pairs + r + sep = 2*num_instr + 1
            mask_len = 2 * num_instr + 1
            label[:mask_len] = [self.tokenizer.pad_token_id] * mask_len

            pos_ids = self.get_pos_ids(
                len(instance), max(0, self.n_positions - len(instance)))

            yield instance, pos_ids, label


class SelectiveCopyDataset(CustomDataset):
    """
    Selective copying: vocabulary V = N ∪ M with |N| = marker_vocab_size numbered tokens
    #1 … #marker_vocab_size and |M| = misc_vocab_size arbitrary filler tokens.

    For content (x_j)_{j=1}^L let i be the maximizer among indices with tokens in N
    (equivalently, the largest j such that x_j ∈ N). The target token is x_{L+1-i}.

    Sequence: <bos> x_1 … x_L <sep> answer <eos> with loss only on answer.
    """

    def __init__(
        self,
        length_range: tuple[int, int],
        max_test_length: int,
        marker_vocab_size: int = 16,
        misc_vocab_size: int = 16,
        add_positional_offset: bool = True,
    ):
        super().__init__(max_test_length + 4, add_positional_offset) # <bos>, <sep>, <eos> and <ans>

        assert marker_vocab_size >= 1 and misc_vocab_size >= 1

        markers = [f"#{k + 1}" for k in range(marker_vocab_size)]
        fillers = [f"m{k}" for k in range(misc_vocab_size)]
        self.tokenizer = customTokenizer(markers + fillers)

        self._marker_vocab_size = marker_vocab_size
        self._misc_vocab_size = misc_vocab_size

        self.range_min, self.range_max = length_range
        self.range_min = max(1, self.range_min)
        self.max_test_length = max_test_length
        assert len(self.tokenizer) - 4 >= marker_vocab_size + misc_vocab_size
        assert (max_test_length >= self.range_max) or (max_test_length == -1)

    def _compute_answer_token_id(self, content: list[int]) -> int:
        """Answer = x_{L+1-i} where i is 1-based index of last position with token in N."""
        last_marker = -1
        for tid in content:
            if tid < self._marker_vocab_size:
                last_marker = tid
        assert last_marker >= 0

        L = len(content) - 1
        return content[L - last_marker]

    def __iter__(self):
        vocab_size = self._marker_vocab_size + self._misc_vocab_size

        while True:
            length = random.randint(self.range_min, self.range_max)
            while True:
                content = [random.randrange(vocab_size) for _ in range(length)]

                valid_instance = False
                for t in content:
                    if t < self._marker_vocab_size: # is marker
                        if t < length:
                            valid_instance = True
                        else:
                            valid_instance = False
                if valid_instance:
                    break

            ans = self._compute_answer_token_id(content)

            instance = [self.tokenizer.bos_token_id]
            instance.extend(content)
            instance.append(self.tokenizer.sep_token_id)
            instance.append(ans)
            instance.append(self.tokenizer.eos_token_id)

            label = deepcopy(instance)
            label[: length + 2] = [self.tokenizer.pad_token_id] * (length + 2)

            pos_ids = self.get_pos_ids(len(instance), self.max_test_length - length)

            yield instance, pos_ids, label


class MKARDataset(CustomDataset):
    """
    Multi-token Key Associative Recall (MKAR).

    For x⃗ ∈ V^L let K = x_{L-k:L} (suffix of length k).
    Let i be the last (largest) start index with x⃗_{i:i+k} = K and i < L−k so the
    match is strictly before the query suffix — then xi+k is well-defined inside x⃗.
    Predict that token xi+k (immediate successor after that earlier occurrence).

    Instances are sampled so K appears non-trivially (at least two occurrences);
    stray collisions remain exponentially rare when |V|^k is large.
    Sequence: <bos> x⃗ <sep> answer <eos> with loss only on answer.
    """

    def __init__(
        self,
        length_range: tuple[int, int],
        max_test_length: int,
        key_len: int = 4,
        vocab_size: int = 128,
        add_positional_offset: bool = True,
    ):
        super().__init__(max_test_length + 4, add_positional_offset)

        assert key_len >= 1
        assert vocab_size >= 2

        self.k = key_len
        vocab = [str(i) for i in range(vocab_size)]
        self.tokenizer = customTokenizer(vocab)

        self.range_min, self.range_max = length_range
        assert self.range_min <= self.range_max

        self.range_min = max(self.range_min, 2 * self.k + 1)
        assert self.range_min <= self.range_max

        self.max_test_length = max_test_length
        assert len(self.tokenizer) - 4 >= vocab_size
        assert (max_test_length >= self.range_max) or (max_test_length == -1)

    @staticmethod
    def _suffix_k(x: list[int], k: int) -> tuple[int, ...]:
        return tuple(x[len(x) - k :])

    def __iter__(self):
        vs = len(self.tokenizer) - 4
        k = self.k

        while True:
            length = random.randint(self.range_min, self.range_max)
            inner_start_hi = length - 2 * k  # inclusive: inner K ends before suffix
            inner_start = random.randint(0, inner_start_hi)
            inner_end = inner_start + k

            tail_start = length - k
            assert inner_end <= tail_start

            K = tuple(random.randrange(vs) for _ in range(k))
            content = [random.randrange(vs) for _ in range(length)]
            content[inner_start:inner_end] = list(K)
            content[tail_start:length] = list(K)

            ans = content[inner_end]

            instance = [self.tokenizer.bos_token_id]
            instance.extend(content)
            instance.append(self.tokenizer.sep_token_id)
            instance.append(ans)
            instance.append(self.tokenizer.eos_token_id)

            label = deepcopy(instance)
            label[: length + 2] = [self.tokenizer.pad_token_id] * (length + 2)

            pos_ids = self.get_pos_ids(len(instance), self.max_test_length - length)

            yield instance, pos_ids, label


# Dyck-2: two bracket types, unbounded nesting. Distinct from the formal-language
# ``d_2`` task, which is *bounded*-depth Dyck-1 (one pair, depth at most 2).
_DYCK2_OPENS = ("(", "[")
_DYCK2_CLOSE_FOR = {"(": ")", "[": "]"}
_DYCK2_OPEN_FOR = {")": "(", "]": "["}


def dyck2_completing_closers(prefix: list[str]) -> list[str]:
    """Unique shortest closer sequence that completes a valid Dyck-2 prefix.

    Raises ``ValueError`` if ``prefix`` is not a valid Dyck-2 prefix (stack would
    go negative or a closer would mismatch the current top).
    """
    stack: list[str] = []
    for tok in prefix:
        if tok in _DYCK2_CLOSE_FOR:
            stack.append(tok)
        elif tok in _DYCK2_OPEN_FOR:
            if not stack or stack[-1] != _DYCK2_OPEN_FOR[tok]:
                raise ValueError(f"invalid Dyck-2 prefix {prefix!r}")
            stack.pop()
        else:
            raise ValueError(f"token {tok!r} is not in the Dyck-2 alphabet")
    return [_DYCK2_CLOSE_FOR[open_tok] for open_tok in reversed(stack)]


def is_dyck2_word(seq: list[str]) -> bool:
    """True iff ``seq`` is a complete Dyck-2 word (valid prefix whose stack is empty)."""
    try:
        return dyck2_completing_closers(seq) == []
    except ValueError:
        return False


def generate_dyck2_prefix(length: int) -> list[str]:
    """Sample a valid Dyck-2 prefix of exact ``length`` by an unbounded stack walk.

    Opening is always legal, so nesting depth is not capped (unlike formal ``D_n``).
    When the stack is empty the next token must be an open; otherwise open/close
    are chosen uniformly.
    """
    if length < 1:
        raise ValueError(f"Dyck-2 prefix length must be >= 1, got {length}")
    stack: list[str] = []
    prefix: list[str] = []
    for _ in range(length):
        if not stack or random.random() < 0.5:
            tok = random.choice(_DYCK2_OPENS)
            prefix.append(tok)
            stack.append(tok)
        else:
            prefix.append(_DYCK2_CLOSE_FOR[stack.pop()])
    return prefix


class Dyck2Dataset(CustomDataset):
    """
    Dyck-2 language of unbounded nesting depth (two bracket types: ``()`` and ``[]``).

    A valid prefix of length ``L`` is generated by a stack random walk that never
    forbids opening, so depth is unbounded. The answer is the unique shortest
    completion -- the closers that empty the stack -- so prefix + answer is always
    a complete Dyck-2 word:

        <bos> ( [ ( ) <sep> ] ) <eos>

    Loss is scored only on the completing closers and ``<eos>``. ``length_range``
    is the prefix length (bracket tokens before ``<sep>``). Worst-case sequence
    length is ``2 L + 3`` (all opens, then all matching closes).
    """

    def __init__(self, length_range: tuple[int, int], max_test_length: int, add_positional_offset: bool = True):
        super().__init__(max_test_length * 2 + 3, add_positional_offset)  # bos, sep, eos + up to L closers

        self.tokenizer = customTokenizer(["(", ")", "[", "]"])
        self.range_min, self.range_max = length_range
        self.range_min = max(1, self.range_min)
        self.max_test_length = max_test_length
        assert (max_test_length >= self.range_max) or (max_test_length == -1)

    def __iter__(self):
        vocab = self.tokenizer.vocab
        pad_id = self.tokenizer.pad_token_id

        while True:
            length = random.randint(self.range_min, self.range_max)
            prefix = generate_dyck2_prefix(length)
            answer = dyck2_completing_closers(prefix)

            instance = [self.tokenizer.bos_token_id]
            instance.extend(vocab[t] for t in prefix)
            instance.append(self.tokenizer.sep_token_id)
            instance.extend(vocab[t] for t in answer)
            instance.append(self.tokenizer.eos_token_id)

            label = deepcopy(instance)
            label[: length + 2] = [pad_id] * (length + 2)  # bos + prefix + sep

            pos_ids = self.get_pos_ids(len(instance), max(0, self.n_positions - len(instance)))

            yield instance, pos_ids, label


class DFA:
    def __init__(self, sigma, q_states, delta, q0, final_states):
        self.sigma = sigma
        self.q_states = q_states
        self.delta = delta
        self.q0 = q0
        self.final_states = final_states

    def __call__(self, string: str) -> bool:
        q_t = self.q0
        for symbol in string:
            q_t = self.delta(q_t, symbol)
        return q_t in self.final_states


class TomitaLanguage(ABC):
    def __init__(self, p: float, q: float):
        self.p = p
        self.q = q
        self.sigma = ["0", "1"]

    @abstractmethod
    def belongs_to_lang(self, seq: str) -> bool:
        raise NotImplementedError

    def generate_string(self, min_length: int, max_length: int) -> str:
        string = ""
        symbols = self.sigma + ["T"]
        while len(string) < max_length:
            symbol = np.random.choice(symbols, p=[self.p, self.q, 1 - (self.p + self.q)])
            if symbol == "T":
                break
            string += str(symbol)
        return string

    def generate_list(self, num: int, min_length: int, max_length: int, leak: bool):
        arr = []
        while len(arr) < num:
            string = self.generate_string(min_length, max_length)
            if not leak and string in arr:
                continue
            if min_length <= len(string) <= max_length and self.belongs_to_lang(string):
                arr.append(string)
        return arr

    def output_generator(self, seq: str):
        return "".join(["y" if self.belongs_to_lang(seq[:i]) else "n" for i in range(1, len(seq) + 1)])

    def training_set_generator(self, num: int, min_size: int, max_size: int, leak: bool):
        input_arr = self.generate_list(num, min_size, max_size, leak)
        output_arr = [self.output_generator(seq) for seq in input_arr]
        return input_arr, output_arr


class Tomita1Language(TomitaLanguage):
    def __init__(self, p: float, q: float):
        super().__init__(p, q)
        self.sigma = ["0", "1"]
        self.q0 = "q0"
        self.dead_states = {"q1"}
        self.dfa = DFA(self.sigma, ["q0", "q1"], self.transition_function, self.q0, {"q0"})

    def transition_function(self, q, s):
        if q == "q0":
            return "q1" if s == "0" else "q0"
        return "q1"

    def get_final_state(self, seq):
        q = self.q0
        for s in seq:
            q = self.transition_function(q, s)
        return q

    def belongs_to_lang(self, seq: str) -> bool:
        return self.dfa(seq)

    def generate_string(self, min_length: int, max_length: int):
        length = np.random.randint(min_length, max_length + 1)
        return "".join(["1" for _ in range(length)])

    def get_legal_characters(self, seq):
        legal_chars = []
        for i in range(len(seq)):
            legal = []
            q_f_0 = self.get_final_state(seq[: i + 1] + "0")
            q_f_1 = self.get_final_state(seq[: i + 1] + "1")
            if q_f_0 not in self.dead_states:
                legal.append("0")
            if q_f_1 not in self.dead_states:
                legal.append("1")
            legal_chars.append(legal)
        return legal_chars

    def output_generator(self, seq):
        output = ""
        for legal in self.get_legal_characters(seq):
            output += "y" if "0" in legal else "n"
            output += "y" if "1" in legal else "n"
        return output


class Tomita2Language(Tomita1Language):
    def __init__(self, p: float, q: float):
        super().__init__(p, q)
        self.q0 = "q0"
        self.dead_states = {"q2"}
        self.dfa = DFA(self.sigma, ["q0", "q1", "q2"], self.transition_function, self.q0, {"q0"})

    def transition_function(self, q, s):
        if q == "q0":
            return "q2" if s == "0" else "q1"
        if q == "q1":
            return "q0" if s == "0" else "q2"
        return "q2"

    def generate_string(self, min_length: int, max_length: int):
        length = (np.random.randint(min_length, max_length) + 1) // 2
        return "".join(["10" for _ in range(length)])


class Tomita3Language(Tomita1Language):
    def __init__(self, p: float, q: float):
        super().__init__(p, q)
        self.q0 = "q0"
        self.dead_states = {"q3", "q4"}
        self.dfa = DFA(self.sigma, ["q0", "q1", "q2", "q3", "q4"], self.transition_function, self.q0, {"q0", "q1", "q2"})

    def transition_function(self, q, s):
        if q == "q0":
            return "q0" if s == "0" else "q1"
        if q == "q1":
            return "q3" if s == "0" else "q0"
        if q == "q2":
            return "q3" if s == "0" else "q1"
        if q == "q3":
            return "q2" if s == "0" else "q4"
        return "q4"

    def generate_string(self, min_length: int, max_length: int):
        length = np.random.randint(min_length, max_length + 1)
        string = ""
        last_toss = None
        last_one_count = 0
        while len(string) != length:
            toss = np.random.choice(["0", "1"])
            if toss == "1":
                char_count = np.random.randint(length - len(string) + 1)
                string += "".join([toss for _ in range(char_count)])
                if last_toss == "0" and char_count != 0:
                    last_one_count = char_count
                else:
                    last_one_count += char_count
            else:
                if last_toss is None or last_one_count % 2 == 0:
                    char_count = np.random.randint(length - len(string) + 1)
                else:
                    choices = np.arange(0, length - len(string) + 1, 2)
                    char_count = np.random.choice(choices)
                string += "".join([toss for _ in range(char_count)])
            if char_count != 0:
                last_toss = toss
        return string


class Tomita4Language(Tomita3Language):
    def __init__(self, p: float, q: float):
        super().__init__(p, q)
        self.q0 = "q0"
        self.dead_states = {"q3"}
        self.dfa = DFA(self.sigma, ["q0", "q1", "q2", "q3"], self.transition_function, self.q0, {"q0", "q1", "q2"})

    def transition_function(self, q, s):
        if q == "q0":
            return "q1" if s == "0" else "q0"
        if q == "q1":
            return "q2" if s == "0" else "q0"
        if q == "q2":
            return "q3" if s == "0" else "q0"
        return "q3"

    def generate_string(self, min_length: int, max_length: int):
        length = np.random.randint(min_length, max_length + 1)
        string = ""
        while len(string) < length:
            toss = np.random.choice(["0", "1"])
            if toss == "0" and len(string) >= 2 and string[-1] == "0" and string[-2] == "0":
                continue
            string += toss
        return string


class Tomita5Language(TomitaLanguage):
    def belongs_to_lang(self, seq: str):
        if seq == "":
            return True
        counter = Counter(seq)
        return (counter["0"] % 2 == 0) and (counter["1"] % 2 == 0)


class Tomita6Language(TomitaLanguage):
    def belongs_to_lang(self, seq: str):
        if seq == "":
            return True
        counter = Counter(seq)
        return abs(counter["0"] - counter["1"]) % 3 == 0


class Tomita7Language(Tomita3Language):
    def __init__(self, p: float, q: float):
        super(Tomita3Language, self).__init__(p, q)
        self.sigma = ["0", "1"]
        self.q0 = "q0"
        self.dead_states = {"q4"}
        self.dfa = DFA(self.sigma, ["q0", "q1", "q2", "q3", "q4"], self.transition_function, self.q0, {"q0", "q1", "q2", "q3"})

    def transition_function(self, q, s):
        if q == "q0":
            return "q0" if s == "0" else "q1"
        if q == "q1":
            return "q2" if s == "0" else "q1"
        if q == "q2":
            return "q2" if s == "0" else "q3"
        if q == "q3":
            return "q4" if s == "0" else "q3"
        return "q4"

    def generate_string(self, min_length: int, max_length: int):
        string = ""
        length = max_length
        num_zeros = np.random.randint(0, length + 1)
        string += "".join(["0" for _ in range(num_zeros)])
        if len(string) == length:
            return string
        num_ones = np.random.randint(0, length - len(string) + 1)
        string += "".join(["1" for _ in range(num_ones)])
        if len(string) == length:
            return string
        num_zeros = np.random.randint(0, length - len(string) + 1)
        string += "".join(["0" for _ in range(num_zeros)])
        if len(string) == length:
            return string
        num_ones = np.random.randint(0, length - len(string) + 1)
        string += "".join(["1" for _ in range(num_ones)])
        return string


class D_nLanguage:
    def __init__(self, n: int) -> None:
        self.n = n
        self.total_tries = 0
        self.std_ratio = 0.1
        self.mean_ratio = 0.75

    def generate_d_n(self, n: int, maxlength: int) -> str:
        if n == 0 or maxlength == 0:
            return ""
        d_n = ""
        while len(d_n) < maxlength:
            length_d_n_min_1 = int(maxlength * self.mean_ratio * (self.std_ratio * np.random.randn() + 1))
            d_n_min_1 = self.generate_d_n(n - 1, length_d_n_min_1)
            d_n += f"a{d_n_min_1}b"
        return d_n

    def generate_string(self, maxlength: int) -> str:
        length = int(maxlength * self.mean_ratio * (self.std_ratio * np.random.randn() + 1))
        return self.generate_d_n(self.n, length)

    def find_depth(self, sequence: str) -> int:
        return sequence.count("a") - sequence.count("b")

    def get_final_state(self, sequence: str) -> str:
        depth = self.find_depth(sequence)
        return "10" if depth == 0 else "01" if depth == self.n else "11"

    def output_generator(self, seq: str) -> str:
        return "".join([self.get_final_state(seq[: i + 1]) for i in range(len(seq))])

    def generate_list(self, num: int, min_length: int, max_length: int):
        input_list = []
        while len(input_list) < num:
            string = self.generate_string(max_length)
            if (string not in input_list) and (min_length <= len(string) <= max_length):
                input_list.append(string)
            else:
                self.total_tries += 1
            if self.total_tries > 20000:
                self.total_tries = 0
                self.mean_ratio -= 0.02
                if self.mean_ratio < 0:
                    break
        return input_list

    def training_set_generator(self, num: int, min_size: int, max_size: int):
        input_arr = self.generate_list(num, min_size, max_size)
        output_arr = [self.output_generator(seq) for seq in input_arr]
        return input_arr, output_arr


class AAStarBBStarLanguage:
    def __init__(self, n: int = 5) -> None:
        letters = "abcdefgh"
        self.possible_chars = letters[:n]
        self.all_chars = self.possible_chars + "T"
        self.char2id = {ch: i for i, ch in enumerate(self.all_chars)}
        self.n_letters = n + 1

    def generate_string(self, min_length: int, max_length: int):
        string = ""
        total_count = max_length - min_length + 1
        for symbol in self.possible_chars:
            count = np.random.randint(total_count) + 1 if total_count > 0 else 0
            symb_count = min_length // (self.n_letters - 1) + count
            string += symb_count * symbol
            total_count -= count
        return string

    def generate_list(self, num: int, min_length: int, max_length: int):
        input_list = []
        while len(input_list) < num:
            string = self.generate_string(min_length, max_length)
            if (string not in input_list) and (min_length <= len(string) <= max_length):
                if self.possible_chars[-1] in string:
                    input_list.append(string)
        return input_list

    def output_generator(self, sequence: str):
        output = "".join([self.all_chars[self.char2id[symbol] + 1] for symbol in sequence])
        return output.upper()

    def training_set_generator(self, num: int, min_size: int, max_size: int):
        input_arr = self.generate_list(num, min_size, max_size)
        output_arr = [self.output_generator(seq) for seq in input_arr]
        return input_arr, output_arr


class AB_D_BC:
    def __init__(self, choices_pre, choices_post, mandatory):
        self.mandatory = mandatory
        self.choices_pre = list(choices_pre)
        self.choices_post = list(choices_post)
        self.pre_map = "1101"
        self.post_map = "0110"

    def generate_string(self, max_length: int):
        pre_length = np.random.randint(0, max_length - 1)
        pre_string = "".join([np.random.choice(self.choices_pre) for _ in range(pre_length)])
        post_length = np.random.randint(0, max_length - pre_length - 1)
        post_string = "".join([np.random.choice(self.choices_post) for _ in range(post_length)])
        return pre_string + self.mandatory + post_string

    def output_generator(self, seq: str):
        split_point = seq.rfind(self.mandatory)
        return "".join([self.pre_map if index < split_point else self.post_map for index in range(len(seq))])

    def generate_list(self, num: int, min_length: int, max_length: int):
        input_list = []
        while len(input_list) < num:
            string = self.generate_string(max_length)
            if (string not in input_list) and (min_length <= len(string) <= max_length):
                input_list.append(string)
        return input_list

    def training_set_generator(self, num: int, min_size: int, max_size: int):
        input_arr = self.generate_list(num, min_size, max_size)
        output_arr = [self.output_generator(seq) for seq in input_arr]
        return input_arr, output_arr


class ZOT_Z_T:
    def __init__(self, choices_pre, choices_post, mandatory):
        self.mandatory = mandatory
        self.choices_pre = list(choices_pre)
        self.choices_post = list(choices_post)

    def generate_string(self, max_length: int):
        pre_length = np.random.randint(0, max_length - 1)
        pre_string = "".join([np.random.choice(self.choices_pre) for _ in range(pre_length)])
        post_length = np.random.randint(0, max_length - pre_length - 1)
        post_string = "".join([np.random.choice(self.choices_post) for _ in range(post_length)])
        return pre_string + self.mandatory + post_string

    def output_generator(self, seq: str):
        is_2_in_end_state = False
        output_str = ""
        for s in seq:
            if s == "1":
                is_2_in_end_state = False
            elif s == "0":
                is_2_in_end_state = True
            if s != "2" or is_2_in_end_state is False:
                output_str += "c"
            else:
                output_str += "e"
        return output_str

    def generate_list(self, num: int, min_length: int, max_length: int):
        input_list = []
        while len(input_list) < num:
            string = self.generate_string(max_length)
            if (string not in input_list) and (min_length <= len(string) <= max_length):
                input_list.append(string)
        return input_list

    def training_set_generator(self, num: int, min_size: int, max_size: int):
        input_arr = self.generate_list(num, min_size, max_size)
        output_arr = [self.output_generator(seq) for seq in input_arr]
        return input_arr, output_arr


class NonStarFreeLanguage(ABC):
    def __init__(self, n: int) -> None:
        letters = ["a", "b", "c", "d", "e", "f", "g", "h"]
        self.sigma = letters[:n]
        self.n_letters = n

    @abstractmethod
    def belongToLang(self, seq: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def generate_string(self, min_length: int, max_length: int) -> str:
        raise NotImplementedError

    def generate_list(self, to_generate_num: int, min_length: int, max_length: int):
        final_list = []
        while len(final_list) < to_generate_num:
            string = self.generate_string(min_length, max_length)
            if min_length <= len(string) <= max_length:
                final_list.append(string)
        return final_list

    def output_generator(self, seq: str) -> str:
        return "".join(["1" if self.belongToLang(seq[:i]) else "0" for i in range(1, len(seq) + 1)])

    def training_set_generator(self, to_generate_num: int, min_size: int, max_size: int):
        input_arr = self.generate_list(to_generate_num, min_size, max_size)
        output_arr = [self.output_generator(seq) for seq in input_arr]
        return input_arr, output_arr


class ABABStarLanguage(NonStarFreeLanguage):
    def __init__(self, n: int = 2) -> None:
        super().__init__(n)

    def belongToLang(self, seq: str):
        sublen = self.n_letters * 2
        if len(seq) % sublen != 0:
            return False
        for i in range(0, len(seq), sublen):
            if seq[i : i + sublen] != "".join(self.sigma + self.sigma):
                return False
        return True

    def generate_string(self, min_length: int, max_length: int):
        sublen = self.n_letters * 2
        num_ababs = (min_length + np.random.randint(max_length - min_length + 1)) // sublen
        return "".join(["".join(self.sigma + self.sigma) for _ in range(num_ababs)])


class AAStarLanguage(NonStarFreeLanguage):
    def __init__(self, n: int) -> None:
        super().__init__(n=1)
        self.n = n

    def belongToLang(self, seq: str):
        req_subseq = "".join([self.sigma[0] for _ in range(self.n)])
        sublen = len(req_subseq)
        if len(seq) % sublen != 0:
            return False
        for i in range(0, len(seq), sublen):
            if seq[i : i + sublen] != req_subseq:
                return False
        return True

    def generate_string(self, min_length: int, max_length: int):
        req_subseq = "".join([self.sigma[0] for _ in range(self.n)])
        sublen = len(req_subseq)
        num_aas = (min_length + np.random.randint(max_length - min_length + 1)) // sublen
        return "".join([req_subseq for _ in range(num_aas)])


class AnStarA2Language(NonStarFreeLanguage):
    def __init__(self, n: int) -> None:
        super().__init__(n=1)
        self.lang = AAStarLanguage(n)

    def generate_string(self, min_length: int, max_length: int):
        return self.lang.generate_string(min_length, max_length) + "aa"

    def belongToLang(self, seq: str):
        if len(seq) < 2 or seq[-2:] != "aa":
            return False
        return self.lang.belongToLang(seq[:-2])


class TomitaCorpus:
    def __init__(self, n: int, lower_window: int, upper_window: int, size: int, unique: bool, leak: bool = False):
        assert 1 <= n <= 7
        avg_len = (lower_window + upper_window) // 2
        p = avg_len / (2 * (1 + avg_len))
        self.unique = unique
        self.leak = leak
        self.lang = globals()[f"Tomita{n}Language"](p, p)
        self.source, self.target = self.generate_data(size, lower_window, upper_window)

    def generate_data(self, size: int, lower_window: int, upper_window: int):
        inputs, outputs = self.lang.training_set_generator(size, lower_window, upper_window, self.leak)
        if self.unique:
            inputs, outputs = zip(*set(zip(inputs, outputs)))
            return list(inputs), list(outputs)
        return inputs, outputs


class StarFreeCorpus:
    def __init__(self, lang: str, num_par: int, lower_window: int, upper_window: int, size: int, unique: bool = False):
        self.lang = globals()[f"{lang}Language"](num_par)
        self.unique = unique
        self.source, self.target = self.generate_data(size, lower_window, upper_window)

    def generate_data(self, size: int, lower_window: int, upper_window: int):
        inputs, outputs = self.lang.training_set_generator(size, lower_window, upper_window)
        if self.unique:
            inputs, outputs = zip(*set(zip(inputs, outputs)))
            return list(inputs), list(outputs)
        return inputs, outputs


class StarFreePostLanguageCorpus:
    def __init__(self, mandatory: str, pre_choices: str, post_choices: str, lower_window: int, upper_window: int, size: int):
        if mandatory == "d":
            self.lang = AB_D_BC(pre_choices, post_choices, mandatory)
        elif mandatory == "0":
            self.lang = ZOT_Z_T(pre_choices, post_choices, mandatory)
        else:
            raise ValueError(f"Unsupported mandatory marker {mandatory!r}")
        self.source, self.target = self.generate_data(size, lower_window, upper_window)

    def generate_data(self, size: int, lower_window: int, upper_window: int):
        inputs, outputs = self.lang.training_set_generator(size, lower_window, upper_window)
        inputs, outputs = zip(*set(zip(inputs, outputs)))
        return list(inputs), list(outputs)


class NonStarFreeCorpus:
    def __init__(self, lang: str, num_par: int, lower_window: int, upper_window: int, size: int, unique: bool = False):
        self.lang = globals()[f"{lang}Language"](num_par)
        self.unique = unique
        self.source, self.target = self.generate_data(size, lower_window, upper_window)

    def generate_data(self, size: int, lower_window: int, upper_window: int):
        inputs, outputs = self.lang.training_set_generator(size, lower_window, upper_window)
        if self.unique:
            inputs, outputs = zip(*set(zip(inputs, outputs)))
            return list(inputs), list(outputs)
        return inputs, outputs


class FormalLanguageDataset(CustomDataset):
    """A formal-language transduction corpus encoded for causal-LM training.

    Both serializations below rely on the autoregressive shift applied by
    `ForCausalLMLoss` and `compute_metrics`, i.e. the logits at position `t` are
    scored against `label[t + 1]`.

    `aligned=True` (the default) matches formal_lang_suite: source and target hold
    one token each per string position, and the model emits target token `t` as soon
    as it has consumed source token `t`:

        input_ids: <bos>   a     a    a    a   <eos>
        label:     <pad> <pad>   0    1    0     1
                            ^ logits after "<bos> a" are scored against target[0]

    The trailing `<eos>` is never scored -- the shift drops its logits -- and exists
    only so the last target token has a position to occupy.

    `aligned=False` is the legacy prompt/answer packing shared with the algorithmic
    tasks, where the whole target follows the source behind a separator:

        input_ids: <bos> a a a a <sep>  0    1    0    1  <eos>
        label:     <pad> ...    <pad>   0    1    0    1  <eos>

    That form asks for strictly more than the language does: the model must also
    retain the source length and count it back down to place `<eos>`, an unbounded
    counter rather than the finite state the language needs. Fixed-state recurrent
    models fit it in-distribution and then fail to extrapolate, so prefer `aligned`
    unless you are deliberately reproducing the old behaviour.
    """

    def __init__(
        self,
        source: list[str],
        target: list[str] | list[list[str]],
        tokenizer: customTokenizer,
        n_positions: int,
        add_positional_offset: bool = True,
        aligned: bool = True,
    ):
        super().__init__(n_positions, add_positional_offset)
        self.source = source
        self.target = target
        self.tokenizer = tokenizer
        self.aligned = aligned
        if aligned:
            mismatch = next(((s, t) for s, t in zip(source, target) if len(s) != len(t)), None)
            if mismatch is not None:
                raise ValueError(
                    "aligned serialization needs one target token per source token, got lengths "
                    f"{len(mismatch[0])} and {len(mismatch[1])}; languages that label each position "
                    "with several characters must have their targets grouped into per-position "
                    "tokens first (see dataset_generators._chunk_targets)"
                )

    def _encode_pair(self, src, tgt):
        src_ids = [self.tokenizer.vocab[token] for token in src]
        tgt_ids = [self.tokenizer.vocab[token] for token in tgt]
        pad_id = self.tokenizer.pad_token_id
        if self.aligned:
            instance = [self.tokenizer.bos_token_id] + src_ids + [self.tokenizer.eos_token_id]
            label = [pad_id, pad_id] + tgt_ids
        else:
            instance = [self.tokenizer.bos_token_id] + src_ids + [self.tokenizer.sep_token_id] + tgt_ids + [self.tokenizer.eos_token_id]
            label = deepcopy(instance)
            label[: len(src_ids) + 2] = [pad_id] * (len(src_ids) + 2)
        pos_ids = self.get_pos_ids(len(instance), self.n_positions - len(instance))
        return instance, pos_ids, label

    def __iter__(self):
        while True:
            idx = random.randrange(len(self.source))
            yield self._encode_pair(self.source[idx], self.target[idx])


class EvalDataset(Dataset):
    """A fixed number of examples drawn eagerly from a streaming dataset."""

    def __init__(self, d: IterableDataset, num_data: int) -> None:
        super().__init__()
        self.source_dataset = d
        self.data = []
        for i, item in enumerate(d):
            if i >= num_data:
                break
            self.data.append(item)

    @property
    def tokenizer(self) -> customTokenizer:
        """The tokenizer that encoded these examples, so they can be decoded again."""
        return self.source_dataset.tokenizer

    @property
    def n_positions(self) -> int:
        return self.source_dataset.n_positions

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
