import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from text import text_to_sequence
from utils import process_meta, pad_1D, pad_2D
import librosa
import tgt
from preprocessors.utils import get_alignment

# Comments: Aligner-related codes 
# from aligner.utils import BetaBinomialInterpolator
# beta_binomial_interpolator = BetaBinomialInterpolator()

def parse_batch(batch):
    sid = torch.from_numpy(batch["sid"]).long().cuda()
    text = torch.from_numpy(batch["text"]).long().cuda()
    mel_target = torch.from_numpy(batch["mel_target"]).float().cuda()
    spec_target = torch.from_numpy(batch["spec_target"]).float().cuda()
    D = torch.from_numpy(batch["D"]).long().cuda()
    log_D = torch.from_numpy(batch["log_D"]).float().cuda()
    f0 = torch.from_numpy(batch["f0"]).float().cuda()
    energy = torch.from_numpy(batch["energy"]).float().cuda()
    src_len = torch.from_numpy(batch["src_len"]).long().cuda()
    spec_len = torch.from_numpy(batch["spec_len"]).long().cuda()
    max_src_len = np.max(batch["src_len"]).astype(np.int32)
    max_spec_len = np.max(batch["spec_len"]).astype(np.int32)
    spec_start_idx = torch.Tensor(batch["spec_start_idx"]).int().cuda()
    wav = torch.from_numpy(np.array(batch["wav"], dtype=np.float32)).cuda()
    # ali_prior = batch["ali_prior"]
    return sid, text, mel_target, spec_target, spec_start_idx, wav, \
            D, log_D, f0, energy, \
            src_len, spec_len, max_src_len, max_spec_len
    '''
    return sid, text, mel_target, ali_prior, spec_target, spec_start_idx, wav, \
            D, log_D, f0, energy, \
            src_len, spec_len, max_src_len, max_spec_len
    '''
    


def prepare_dataloader(data_path, filename, batch_size, shuffle=True, num_workers=2, meta_learning=False, seed=0, val=False):
    dataset = TextMelDataset(data_path, filename, val=val)
    if meta_learning:
        sampler = MetaBatchSampler(dataset.sid_to_indexes, batch_size, seed=seed)
    else:
        sampler = None
    shuffle = shuffle if sampler is None else None
    if meta_learning:
        loader = DataLoader(dataset, batch_sampler=sampler, 
                        collate_fn=dataset.collate_fn, num_workers=num_workers, pin_memory=True) 
    else:
        loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, shuffle=shuffle, 
                        collate_fn=dataset.collate_fn, drop_last=True, num_workers=num_workers) 
    return loader


def replace_outlier(values, max_v, min_v):
    values = np.where(values<max_v, values, max_v)
    values = np.where(values>min_v, values, min_v)
    return values


def norm_mean_std(x, mean, std):
    x = (x - mean) / std
    return x


class TextMelDataset(Dataset):
    def __init__(self, data_path, filename="train.txt", val=False):
        self.data_path = data_path
        # self.basename, self.text, self.sid = process_meta(data_path, os.path.join(data_path, filename))
        if val is False:
            self.basename, self.text, self.sid = process_meta(data_path, os.path.join(data_path, filename))
        else:
            self.basename, self.text, self.sid = process_meta(data_path, filename)

        self.sid_dict = self.create_speaker_table(self.sid)

        self.num_melbins = 32

        with open(os.path.join(data_path, 'stats.json')) as f:
            data = f.read()
        stats_config = json.loads(data)
        self.f0_stat = stats_config["f0_stat"] # max, min, mean, std
        self.energy_stat = stats_config["energy_stat"] # max, min, mean, std

        self.sampling_rate = 16000
        self.hop_length = 256
        
        self.create_sid_to_index()
        print('Speaker Num :{}'.format(len(self.sid_dict)))

        self.val = val
    
    def create_speaker_table(self, sids):
        speaker_ids = np.sort(np.unique(sids))
        d = {speaker_ids[i]: i for i in range(len(speaker_ids))}
        return d

    def create_sid_to_index(self):
        _sid_to_indexes = {} 
        # for keeping instance indexes with the same speaker ids
        for i, sid in enumerate(self.sid):
            if sid in _sid_to_indexes:
                _sid_to_indexes[sid].append(i)
            else:
                _sid_to_indexes[sid] = [i]
        self.sid_to_indexes = _sid_to_indexes

    def load_audio(self, sid, basename):
        tg_path = os.path.join(self.data_path, "TextGrid", str(sid), "{}.TextGrid".format(basename))
        textgrid = tgt.io.read_textgrid(tg_path)
        phone, duration, start, end = get_alignment(textgrid.get_tier_by_name('phones'), self.sampling_rate, self.hop_length)

        wav_path = os.path.join(self.data_path, "../wav16", str(sid), "{}.wav".format(basename))
        wav, _ = librosa.load(wav_path, sr=self.sampling_rate)
        wav = wav[int(self.sampling_rate*start):int(self.sampling_rate*end)].astype(np.float32)
        return wav

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        sid = self.sid_dict[self.sid[idx]]
        phone = np.array(text_to_sequence(self.text[idx], []))

        mel_path = os.path.join(
            self.data_path, "mel", "libritts-mel-{}.npy".format(basename))
        mel_target = np.load(mel_path)
        if mel_target.shape[0] <= self.num_melbins: # mel length should be longer than mel-window-size 
            return self.__getitem__(idx-1) 

        spec_path = os.path.join(
            self.data_path, "spectrogram", "libritts-spec-{}.npy".format(basename))
        spec_target = np.load(spec_path)
        if spec_target.shape[0] <= self.num_melbins: # mel length should be longer than mel-window-size 
            return self.__getitem__(idx-1) 
        
        if mel_target.shape[0] != spec_target.shape[0]:
            return self.__getitem__(idx-1) 

        real_sid, _, _, _ = basename.split("_")
        wav = self.load_audio(real_sid, basename)
        if spec_target.shape[0] == self.num_melbins:
            spec_start_idx = 0
        else:
            spec_start_idx = np.random.randint(spec_target.shape[0] - self.num_melbins)
        if self.val == False:
            wav = wav[spec_start_idx * self.hop_length:(spec_start_idx + self.num_melbins) * self.hop_length]
        '''
        align_prior_matrix = torch.from_numpy(beta_binomial_interpolator(spec_target.shape[0], phone.shape[0]))
        '''
        
        D_path = os.path.join(
            self.data_path, "alignment", "libritts-ali-{}.npy".format(basename))
        D = np.load(D_path)
        f0_path = os.path.join(
            self.data_path, "f0_avg", "libritts-f0-{}.npy".format(basename))
        f0 = np.load(f0_path)
        f0 = replace_outlier(f0,  self.f0_stat[0], self.f0_stat[1])
        f0 = norm_mean_std(f0, self.f0_stat[2], self.f0_stat[3])
        energy_path = os.path.join(
            self.data_path, "energy_avg", "libritts-energy-{}.npy".format(basename))
        energy = np.load(energy_path)
        energy = replace_outlier(energy, self.energy_stat[0], self.energy_stat[1])
        energy = norm_mean_std(energy, self.energy_stat[2], self.energy_stat[3])
        
        sample = {"id": basename,
                "sid": sid,
                "text": phone,
                "mel_target": mel_target,
                # "ali_prior": align_prior_matrix,
                "spec_target": spec_target, # (melbin, 513) (* mel-spectrogram: (melbin, 80))
                "spec_start_idx": spec_start_idx,
                "wav": wav,
                "D": D,
                "f0": f0,
                "energy": energy}
                
        return sample

    '''
    def pad_prior(self, x, max_spec, max_text):
        # x: (spec_len, text_len)
        spec_len, text_len = x.shape
        m = torch.nn.ZeroPad2d((0, (max_spec - spec_len), 0, (max_text - spec_len)))
        return m(x)
    '''

    def reprocess(self, batch, cut_list):
        ids = [batch[ind]["id"] for ind in cut_list]
        sids = [batch[ind]["sid"] for ind in cut_list]
        texts = [batch[ind]["text"] for ind in cut_list]
        mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
        spec_targets = [batch[ind]["spec_target"] for ind in cut_list]
        spec_start_idxes = [batch[ind]["spec_start_idx"] for ind in cut_list]
        wavs = [batch[ind]["wav"] for ind in cut_list]
        Ds = [batch[ind]["D"] for ind in cut_list]
        f0s = [batch[ind]["f0"] for ind in cut_list]
        energies = [batch[ind]["energy"] for ind in cut_list]
        for text, D, id_ in zip(texts, Ds, ids):
            if len(text) != len(D):
                print(text, text.shape, D, D.shape, id_)
        length_text = np.array(list())
        for text in texts:
            length_text = np.append(length_text, text.shape[0])

        length_spec = np.array(list())
        for spec in spec_targets:
            length_spec = np.append(length_spec, spec.shape[0])
        
        texts = pad_1D(texts)
        Ds = pad_1D(Ds)
        mel_targets = pad_2D(mel_targets)
        spec_targets = pad_2D(spec_targets)
        f0s = pad_1D(f0s)
        energies = pad_1D(energies)
        log_Ds = np.log(Ds + 1.)

        '''
        max_text = int(np.max(length_text))
        max_spec = int(np.max(length_spec))
        ali_prior = torch.stack([self.pad_prior(batch[ind]["ali_prior"], max_spec, max_text) for ind in cut_list], dim=0) # [B, max(spec_len), max(text_len)]
        '''
        out = {"id": ids,
               "sid": np.array(sids),
               "text": texts,
               "mel_target": mel_targets,
               # "ali_prior": ali_prior,
               "spec_target": spec_targets,
               "spec_start_idx": spec_start_idxes,
               "wav": wavs,
               "D": Ds,
               "log_D": log_Ds,
               "f0": f0s,
               "energy": energies,
               "src_len": length_text,
               "spec_len": length_spec}
        
        return out

    def collate_fn(self, batch):
        len_arr = np.array([d["text"].shape[0] for d in batch])
        index_arr = np.argsort(-len_arr)
        output = self.reprocess(batch, index_arr)

        return output


class MetaBatchSampler():
    def __init__(self, sid_to_idx, batch_size, max_iter=100000, seed=0):
        # iterdict contains {sid: [idx1, idx2, ...]}
        np.random.seed(seed)

        self.sids = list(sid_to_idx.keys())
        np.random.shuffle(self.sids)

        self.sid_to_idx = sid_to_idx
        self.batch_size = batch_size
        self.max_iter = max_iter       
        
    def __iter__(self):
        for _ in range(self.max_iter):
            selected_sids = np.random.choice(self.sids, self.batch_size, replace=False)
            batch = []
            for sid in selected_sids:
                idx = np.random.choice(self.sid_to_idx[sid], 1)[0]
                batch.append(idx)

            assert len(batch) == self.batch_size
            yield batch

    def __len__(self):
        return self.max_iter