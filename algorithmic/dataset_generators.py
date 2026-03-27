import random
import string

from copy import deepcopy
from collections import Counter

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
            offset = random.randint(0, max_offset)
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
                 query_fraction: float = 0.2, monoid_type: str = "parity", monoid_n: int = 2):
        """
        MQAR Word Problem dataset.

        Args:
            tokenizer: customTokenizer whose vocab is [k0..k_{K-1}, m0..m_{M-1}] + specials.
            length_range: (min, max) for the content length (2T + Q).
            max_test_length: max content length for position ID offset (-1 for eval).
            key_size: |K|, number of distinct keys.
            monoid_size: |M|, number of monoid elements.
            query_fraction: fraction of content length devoted to queries.
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
        self.query_fraction = query_fraction


        # Key token IDs: 0..key_size-1
        # Monoid token IDs: key_size..key_size+monoid_size-1
        self.monoid_token_offset = key_size

        # Validate constraints
        assert 0 < query_fraction < 1, "query_fraction must be in (0, 1)"
        # Ensure we can always form valid instances at range_min
        self._validate_length(self.range_min)

    def _derive_T_Q(self, length: int):
        """Derive T (num update pairs) and Q (num queries) from content length."""
        Q = max(1, round(self.query_fraction * length))
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


class EvalDataset(Dataset):
    def __init__(self, d: IterableDataset, num_data: int) -> None:
        super().__init__()
        self.data = []
        for i, item in enumerate(d):
            if i >= num_data:
                break
            self.data.append(item)

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)