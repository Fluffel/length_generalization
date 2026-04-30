import random
import string

from copy import deepcopy
from collections import Counter

try:
    from utils import RunConfig
except ImportError:
    # When imported as `algorithmic.dataset_generators` with repo root on sys.path
    from algorithmic.utils import RunConfig
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


def build_datasets(run_config: RunConfig):
    train_length_range = run_config.train_length_range
    test_length_ranges = run_config.test_length_ranges
    max_test_length = test_length_ranges[-1][1]
    test_num = run_config.test_num
    task = run_config.task

    match task:
        case "bin_majority":
            train_dataset = BinaryMajorityDataset(train_length_range, max_test_length)
            test_dataset = {
                f"len{r[0]}-{r[1]}": EvalDataset(
                    BinaryMajorityDataset(r, max_test_length, add_positional_offset=False),
                    test_num,
                )
                for r in test_length_ranges
            }
        case "majority":
            train_dataset = MajorityDataset(train_length_range, max_test_length)
            test_dataset = {
                f"len{r[0]}-{r[1]}": EvalDataset(MajorityDataset(r, max_test_length, add_positional_offset=False), test_num)
                for r in test_length_ranges
            }
        case "bin_majority_interleave":
            train_dataset = BinaryMajorityInterleaveDataset(train_length_range, max_test_length, period=3)
            test_dataset = {
                f"len{r[0]}-{r[1]}": EvalDataset(
                    BinaryMajorityInterleaveDataset(r, max_test_length, period=3, add_positional_offset=False),
                    test_num,
                )
                for r in test_length_ranges
            }
        case "unique_copy":
            train_dataset = UniqueCopyDataset(train_length_range, max_test_length)
            test_dataset = {
                f"len{r[0]}-{r[1]}": EvalDataset(UniqueCopyDataset(r, max_test_length, add_positional_offset=False), test_num)
                for r in test_length_ranges
            }
        case "repeat_copy":
            train_dataset = RepeatCopyDataset(train_length_range, max_test_length)
            test_dataset = {
                f"len{r[0]}-{r[1]}": EvalDataset(RepeatCopyDataset(r, max_test_length, add_positional_offset=False), test_num)
                for r in test_length_ranges
            }
        case "sort":
            train_dataset = SortDataset(train_length_range, max_test_length)
            test_dataset = {
                f"len{r[0]}-{r[1]}": EvalDataset(SortDataset(r, max_test_length, add_positional_offset=False), test_num)
                for r in test_length_ranges
            }
        case "parity":
            train_dataset = ParityDataset(train_length_range, max_test_length)
            test_dataset = {
                f"len{r[0]}-{r[1]}": EvalDataset(ParityDataset(r, max_test_length, add_positional_offset=False), test_num)
                for r in test_length_ranges
            }
        case "addition":
            train_dataset = AdditionDataset(train_length_range, max_test_length)
            test_dataset = {
                f"len{r[0]}-{r[1]}": EvalDataset(AdditionDataset(r, max_test_length, add_positional_offset=False), test_num)
                for r in test_length_ranges
            }
        case "mqar":
            train_dataset = MQARWordProblemDataset(
                train_length_range,
                max_test_length,
                key_size=run_config.key_size,
                query_fraction=run_config.query_fraction,
                monoid_type=run_config.monoid,
                monoid_n=run_config.monoid_n,
            )
            test_dataset = {
                f"len{r[0]}-{r[1]}": EvalDataset(
                    MQARWordProblemDataset(
                        r,
                        max_test_length,
                        add_positional_offset=False,
                        key_size=run_config.key_size,
                        query_fraction=run_config.query_fraction,
                        monoid_type=run_config.monoid,
                        monoid_n=run_config.monoid_n,
                    ),
                    test_num,
                )
                for r in test_length_ranges
            }
        case "flipflop":
            train_dataset = FlipFlopDataset(train_length_range, max_test_length)
            test_dataset = {
                f"len{r[0]}-{r[1]}": EvalDataset(FlipFlopDataset(r, max_test_length, add_positional_offset=False), test_num)
                for r in test_length_ranges
            }
        case "selective_copy":
            train_dataset = SelectiveCopyDataset(train_length_range, max_test_length)
            test_dataset = {
                f"len{r[0]}-{r[1]}": EvalDataset(
                    SelectiveCopyDataset(r, max_test_length, add_positional_offset=False),
                    test_num,
                )
                for r in test_length_ranges
            }
        case "mkar":
            train_dataset = MKARDataset(train_length_range, max_test_length)
            test_dataset = {
                f"len{r[0]}-{r[1]}": EvalDataset(
                    MKARDataset(r, max_test_length, add_positional_offset=False),
                    test_num,
                )
                for r in test_length_ranges
            }
        case _:
            raise ValueError(f"Unknown task {task!r}")

    return train_dataset, test_dataset, train_length_range, test_length_ranges

