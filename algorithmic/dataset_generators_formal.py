import copy
import random
from abc import ABC, abstractmethod
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

try:
    from utils import RunConfig
except ImportError:
    from algorithmic.utils import RunConfig


class customTokenizer:
    def __init__(self, vocab: list[str]):
        normal_tkn_num = len(vocab)

        self.bos_token = "<bos>"
        self.sep_token = "<sep>"
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.bos_token_id = normal_tkn_num
        self.sep_token_id = normal_tkn_num + 1
        self.eos_token_id = normal_tkn_num + 2
        self.pad_token_id = normal_tkn_num + 3
        self.special_token_ids = [
            self.bos_token_id,
            self.sep_token_id,
            self.eos_token_id,
            self.pad_token_id,
        ]
        self.special_tokens = [
            self.bos_token,
            self.sep_token,
            self.eos_token,
            self.pad_token,
        ]
        assert all(t not in vocab for t in self.special_tokens)

        self.vocab = {t: i for i, t in enumerate(vocab)}
        self.vocab[self.bos_token] = self.bos_token_id
        self.vocab[self.sep_token] = self.sep_token_id
        self.vocab[self.eos_token] = self.eos_token_id
        self.vocab[self.pad_token] = self.pad_token_id

        self.vocab_inv = {v: k for k, v in self.vocab.items()}
        self.padding_side = "right"

    def __call__(self, strings: list[str] | str, **kwargs):
        if isinstance(strings, str):
            strings = [strings]
        ids = []
        strings = [s.split(" ") for s in strings]
        max_len = max(map(lambda x: len(x), strings))
        for s in strings:
            ids.append(list(map(lambda x: self.vocab[x], s)) + [self.pad_token_id] * (max_len - len(s)))
        return {"input_ids": torch.LongTensor(ids)}

    def convert_ids_to_tokens(self, ids: list[int], rm_special: bool = False):
        if rm_special:
            return [self.vocab_inv[i] for i in ids if i not in self.special_token_ids]
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

    def get_pos_ids(self, instance_length: int, max_offset: int):
        offset = 0
        if self._add_positional_offset:
            offset = random.randint(0, max(0, max_offset))
        return list(range(offset, instance_length + offset))


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
    def __init__(
        self,
        source: list[str],
        target: list[str],
        tokenizer: customTokenizer,
        n_positions: int,
        add_positional_offset: bool = True,
    ):
        super().__init__(n_positions, add_positional_offset)
        self.source = source
        self.target = target
        self.tokenizer = tokenizer

    def _encode_pair(self, src: str, tgt: str):
        src_ids = [self.tokenizer.vocab[ch] for ch in src]
        tgt_ids = [self.tokenizer.vocab[ch] for ch in tgt]
        instance = [self.tokenizer.bos_token_id] + src_ids + [self.tokenizer.sep_token_id] + tgt_ids + [self.tokenizer.eos_token_id]
        label = copy.deepcopy(instance)
        label[: len(src_ids) + 2] = [self.tokenizer.pad_token_id] * (len(src_ids) + 2)
        pos_ids = self.get_pos_ids(len(instance), self.n_positions - len(instance))
        return instance, pos_ids, label

    def __iter__(self):
        while True:
            idx = random.randrange(len(self.source))
            yield self._encode_pair(self.source[idx], self.target[idx])


def _make_tokenizer_and_n_positions(train_source, train_target, test_bins):
    all_src = list(train_source)
    all_tgt = list(train_target)
    for bin_corpus in test_bins:
        all_src.extend(bin_corpus.source)
        all_tgt.extend(bin_corpus.target)

    vocab = sorted(set("".join(all_src) + "".join(all_tgt)))
    tokenizer = customTokenizer(vocab)
    n_positions = max(1 + len(s) + 1 + len(t) + 1 for s, t in zip(all_src, all_tgt))
    return tokenizer, n_positions


def build_datasets(run_config: RunConfig):
    train_length_range = run_config.train_length_range
    test_length_ranges = run_config.test_length_ranges
    test_num = run_config.test_num
    task = run_config.task

    train_num = max(4 * test_num, 1000)
    lower_window, upper_window = train_length_range

    if task.startswith("tomita_"):
        n = int(task.split("_")[1])
        train_corpus = TomitaCorpus(n, lower_window, upper_window, train_num, unique=False, leak=True)
        test_bins = [TomitaCorpus(n, r[0], r[1], test_num, unique=True, leak=True) for r in test_length_ranges]
    elif task in {"abab_star", "aa_star", "an_star_a2"}:
        lang_map = {
            "abab_star": ("ABABStar", 2),
            "aa_star": ("AAStar", 2),
            "an_star_a2": ("AnStarA2", 2),
        }
        lang_name, num_par = lang_map[task]
        train_corpus = NonStarFreeCorpus(lang_name, num_par, lower_window, upper_window, train_num)
        test_bins = [NonStarFreeCorpus(lang_name, num_par, r[0], r[1], test_num, unique=True) for r in test_length_ranges]
    elif task.startswith("d_"):
        n = int(task.split("_")[1])
        train_corpus = StarFreeCorpus("D_n", n, lower_window, upper_window, train_num, unique=False)
        test_bins = [StarFreeCorpus("D_n", n, r[0], r[1], test_num, unique=False) for r in test_length_ranges]
    elif task == "aa_star_bb_star":
        train_corpus = StarFreeCorpus("AAStarBBStar", 5, lower_window, upper_window, train_num, unique=True)
        test_bins = [StarFreeCorpus("AAStarBBStar", 5, r[0], r[1], test_num, unique=True) for r in test_length_ranges]
    elif task == "ab_star_d_bc_star":
        train_corpus = StarFreePostLanguageCorpus("d", "ab", "bc", lower_window, upper_window, train_num)
        test_bins = [StarFreePostLanguageCorpus("d", "ab", "bc", r[0], r[1], test_num) for r in test_length_ranges]
    elif task == "012_star_0_2_star":
        train_corpus = StarFreePostLanguageCorpus("0", "012", "2", lower_window, upper_window, train_num)
        test_bins = [StarFreePostLanguageCorpus("0", "012", "2", r[0], r[1], test_num) for r in test_length_ranges]
    else:
        raise ValueError(f"Unknown formal task {task!r}")

    tokenizer, n_positions = _make_tokenizer_and_n_positions(
        train_corpus.source,
        train_corpus.target,
        test_bins,
    )
    train_dataset = FormalLanguageDataset(train_corpus.source, train_corpus.target, tokenizer, n_positions)
    test_dataset = {
        f"len{r[0]}-{r[1]}": EvalDataset(
            FormalLanguageDataset(bin_corpus.source, bin_corpus.target, tokenizer, n_positions, add_positional_offset=False),
            min(test_num, len(bin_corpus.source)),
        )
        for r, bin_corpus in zip(test_length_ranges, test_bins)
    }
    return train_dataset, test_dataset, train_length_range, test_length_ranges

