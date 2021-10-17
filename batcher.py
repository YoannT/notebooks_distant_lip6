import pprint
from collections import defaultdict
from itertools import chain

import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse, csr_matrix, vstack
from torch.utils.data import BatchSampler

from torch.utils.data import DataLoader, BatchSampler

def flatten_array(array, mask=None):
    """
    Flatten array to get the list of active entries
    If not mask is provided, it's just array.view(-1)
    If a mask is given, then it is array[mask]
    If the array or the mask are sparse, some optimizations are possible, justifying this function

    Parameters
    ----------
    array: scipy.sparse.spmatrix or np.ndarray or torch.Tensor
    mask: scipy.sparse.spmatrix or np.ndarray or torch.Tensor

    Returns
    -------
    np.ndarray or torch.Tensor
    """
    if issparse(array):
        array = array.tocsr(copy=True)
        col_bis = array.copy()
        col_bis.data = np.ones(len(array.data), dtype=bool)
        if mask is not None and issparse(mask):
            if 0 in mask.shape:
                return np.asarray([], dtype=array.dtype)
            res = array[mask]
            # If empty mask, scipy returns a sparse matrix: we use toarray to densify
            if hasattr(res, 'toarray'):
                return res.toarray().reshape(-1)
            # else, scipy returns a 2d matrix, we use asarray to densify
            return np.asarray(res).reshape(-1)
        array = array.toarray()
        if mask is not None:
            mask = as_numpy_array(mask)
    if isinstance(array, (list, tuple)):
        if mask is None:
            return array
        array = np.asarray(array)
    if isinstance(array, np.ndarray):
        if mask is not None:
            if not isinstance(array, np.ndarray):
                raise Exception(f"Mask type {repr(type(mask))} should be the same as array type {repr(type(array))}")
            return array[mask]
        else:
            return array.reshape(-1)
    elif torch.is_tensor(array):
        if mask is not None:
            if not torch.is_tensor(mask):
                raise Exception(f"Mask type {repr(type(mask))} should be the same as array type {repr(type(array))}")
            return array[mask]
        else:
            return array.reshape(-1)
    else:
        raise Exception(f"Unrecognized array type {repr(type(array))} during array flattening (mask type is {repr(type(mask))}')")


def factorize(values, mask=None, reference_values=None, freeze_reference=True):
    """
    Express values in "col" as row numbers in a reference list of values
    The reference values list is the deduplicated concatenation of preferred_unique_values (if not None) and col

    Ex:
    >>> factorize(["A", "B", "C", "D"], None, ["D", "B", "C", "A", "E"])
    ... [3, 2, 1, 0], None, ["D", "B", "C", "A", "E"]
    >>> factorize(["A", "B", "C", "D"], None, None)
    ... [0, 1, 2, 3], None, ["A", "B", "C", "D"]

    Parameters
    ----------
    col: np.ndarray or scipy.sparse.spmatrix or torch.Tensor or list of (np.ndarray or scipy.sparse.spmatrix or torch.Tensor)
        values to factorize
    mask: np.ndarray or scipy.sparse.spmatrix or torch.Tensor or list of (np.ndarray or scipy.sparse.spmatrix or torch.Tensor) or None
        optional mask on col, useful for multiple dimension values arrays
    freeze_reference: bool
        Should we throw out values out of reference values (if given).
        Then we need a mask to mark those rows as disabled
        TODO: handle cases when a mask is not given
    reference_values: np.ndarray or scipy.sparse.spmatrix or torch.Tensor or list or None
        If given, any value in col that is not in prefered_unique_values will be thrown out
        and the mask will be updated to be False for this value

    Returns
    -------
    col, updated mask, reference values
    """
    if isinstance(values, list) and not hasattr(values[0], '__len__'):
        values = np.asarray(values)
    return_as_list = isinstance(values, list)
    all_values = values if isinstance(values, list) else [values]
    del values
    all_masks = mask if isinstance(mask, list) else [None for _ in all_values] if mask is None else [mask]
    del mask

    assert len(all_values) == len(all_masks), "Mask and values lists must have the same length"

    if reference_values is None:
        freeze_reference = False

    all_flat_values = []
    for values, mask in zip(all_values, all_masks):
        assert (
              (isinstance(mask, np.ndarray) and isinstance(values, np.ndarray)) or
              (issparse(mask) and issparse(values)) or
              (torch.is_tensor(mask) and torch.is_tensor(values)) or
              (mask is None and (isinstance(values, (list, tuple, np.ndarray)) or issparse(values) or torch.is_tensor(values)))), (
            f"values and (optional mask) should be of same type torch.tensor, numpy.ndarray or scipy.sparse.spmatrix. Given types are values: {repr(type(values))} and mask: {repr(type(mask))}")
        all_flat_values.append(flatten_array(values, mask))
        # return all_values[0], all_masks[0], all_values[0].tocsr(copy=True).data if hasattr(all_values[0], 'tocsr') else all_values#col.tocsr(copy=True).data if hasattr(col, 'tocsr')

    device = all_flat_values[0].device if torch.is_tensor(all_flat_values[0]) else None
    if sum(len(vec) for vec in all_flat_values) == 0:
        relative_values = all_flat_values[0]
        unique_values = all_flat_values[0]
    # elif torch.is_tensor(all_flat_values[0]):
    #     device = all_flat_values[0].device
    #     if reference_values is None:
    #         unique_values, relative_values = torch.unique(torch.cat(all_flat_values).unsqueeze(0), dim=1, sorted=False, return_inverse=True)
    #     elif freeze_reference:
    #         relative_values, unique_values = torch.unique(torch.cat((reference_values, *all_flat_values)).unsqueeze(0), dim=1, sorted=False, return_inverse=True)[1], reference_values
    #         print(all_flat_values[0], "VS", reference_values, "=>", relative_values)
    #     else:
    #         unique_values, relative_values = torch.unique(torch.cat((reference_values, *all_flat_values)).unsqueeze(0), dim=1, sorted=False, return_inverse=True)
    #     relative_values = relative_values.squeeze(0)
    else:
        was_tensor = False
        if torch.is_tensor(all_flat_values[0]):
            was_tensor = True
            all_flat_values = [as_numpy_array(v) for v in all_flat_values]
            reference_values = as_numpy_array(reference_values)
        if reference_values is None:
            relative_values, unique_values = pd.factorize(np.concatenate(all_flat_values))
        elif freeze_reference:
            relative_values, unique_values = pd.factorize(np.concatenate((reference_values, *all_flat_values)))[0], reference_values
        else:
            relative_values, unique_values = pd.factorize(np.concatenate((reference_values, *all_flat_values)))
        if was_tensor:
            relative_values = as_tensor(relative_values, device=device)
            unique_values = as_tensor(unique_values, device=device)

    if freeze_reference:
        all_unk_masks = relative_values < len(reference_values)
    else:
        all_unk_masks = None

    offset = 0 if reference_values is None else len(reference_values)
    new_flat_values = []
    new_flat_values = []
    unk_masks = []
    for flat_values in all_flat_values:
        indexer = slice(offset, offset + len(flat_values))
        new_flat_values.append(relative_values[indexer])
        unk_masks.append(all_unk_masks[indexer] if all_unk_masks is not None else None)
        offset = indexer.stop
    all_flat_values = new_flat_values
    del new_flat_values

    if freeze_reference:
        unique_values = unique_values[:len(reference_values)]
    new_values = []
    new_masks = []
    for values, mask, flat_relative_values, unk_mask in zip(all_values, all_masks, all_flat_values, unk_masks):
        if issparse(values):
            new_data = flat_relative_values + 1
            if unk_mask is not None:
                new_data[~unk_mask] = 0
            if mask is None:
                values = values.tocsr(copy=True)
                values.data = new_data
                values.eliminate_zeros()
                new_mask = values.copy()
                values.data -= 1
                new_mask.data = np.ones(len(new_mask.data), dtype=bool)
            else:
                values = mask.tocsr(copy=True)
                values.data = new_data
                values.eliminate_zeros()
                new_mask = values.copy()
                values.data -= 1
                new_mask.data = np.ones(len(new_mask.data), dtype=bool)
            new_values.append(values.tolil())
            new_masks.append(new_mask.tolil())
        elif isinstance(values, (list, tuple)):
            mask = unk_mask
            if mask is not None:
                values = [v for v, valid in zip(flat_relative_values, mask) if valid]
                new_values.append(values)
                new_masks.append(None)
            else:
                values = list(flat_relative_values)
                new_values.append(values)
                new_masks.append(None)
        elif isinstance(values, np.ndarray):
            new_mask = mask
            if freeze_reference:
                if mask is None:
                    new_mask = unk_mask.reshape(values.shape)
                else:
                    new_mask = mask.copy()
                    mask[mask] = unk_mask
            if mask is not None:
                values = np.zeros(values.shape, dtype=int)
                values[mask] = flat_relative_values[unk_mask] if unk_mask is not None else flat_relative_values
                new_values.append(values)
                new_masks.append(new_mask)
            else:
                values = flat_relative_values.reshape(values.shape)
                new_values.append(values)
                new_masks.append(new_mask)
        else:  # torch
            new_mask = mask
            if freeze_reference:
                if mask is None:
                    new_mask = unk_mask.view(*values.shape)
                else:
                    new_mask = mask.clone()
                    mask[mask] = unk_mask
            if mask is not None:
                values = torch.zeros(values.shape, dtype=torch.long, device=device)
                values[mask] = flat_relative_values[unk_mask] if unk_mask is not None else flat_relative_values
                new_values.append(values)
                new_masks.append(mask)
            else:
                values = flat_relative_values.view(*values.shape)
                new_values.append(values)
                new_masks.append(new_mask)
    if return_as_list:
        return new_values, new_masks, unique_values
    return new_values[0], new_masks[0], unique_values


def get_deduplicator(values):
    if isinstance(values, np.ndarray):
        perm = values.argsort()
        sorted_values = values[perm]
        mask = np.ones_like(values, dtype=bool)
        mask[1:] = sorted_values[1:] != sorted_values[:-1]
        return perm[mask]
    elif torch.is_tensor(values):
        perm = values.argsort()
        sorted_values = values[perm]
        mask = torch.ones_like(values, dtype=torch.bool)
        mask[1:] = sorted_values[1:] != sorted_values[:-1]
        return perm[mask]
    else:
        raise Exception()


def index_slice(values, indices):
    if issparse(values):
        return values[indices]
    if hasattr(values, 'shape'):
        return values[indices]
    elif hasattr(indices, 'shape'):
        return as_array(values, type(indices), device=getattr(indices, 'device', None))[indices]
    else:
        return type(values)(as_numpy_array(values)[as_numpy_array(indices)])


def as_numpy_array(array, dtype=None):
    if array is None:
        return None
    if isinstance(array, np.ndarray):
        pass
    elif hasattr(array, 'toarray'):
        if dtype is None or np.issubdtype(array.dtype, dtype):
            return array.toarray()
        return array.astype(dtype).toarray()
    elif torch.is_tensor(array):
        array = array.cpu().numpy()
    else:
        array = np.asarray(array, dtype=dtype)
        return array
    if dtype is None or np.issubdtype(array.dtype, dtype):
        return array
    return array.astype(dtype)


def as_tensor(array, device=None, dtype=None):
    if array is None:
        return None
    if torch.is_tensor(array):
        device = device if device is not None else torch.device('cpu')
        return array.to(device)
    elif isinstance(array, np.ndarray):
        return torch.as_tensor(array, device=device, dtype=dtype)
    elif hasattr(array, 'toarray'):
        return torch.as_tensor(array.toarray(), device=device, dtype=dtype)
    else:
        return torch.as_tensor(array, device=device, dtype=dtype)


def as_array(array, t, device, dtype=None):
    if array is None:
        return None
    if issubclass(t, torch.Tensor):
        return as_tensor(array, device=device, dtype=dtype)
    else:
        return as_numpy_array(array, dtype=dtype)


def concat(arrays):
    if isinstance(arrays[0], np.ndarray):
        return np.concatenate(arrays)
    elif torch.is_tensor(arrays[0]):
        return torch.cat(arrays)
    elif issparse(arrays[0]):
        max_width = max(a.shape[1] for a in arrays)
        for a in arrays:
            a.resize(a.shape[0], max_width)
        return vstack(arrays, format='csr')
    elif isinstance(arrays[0], (tuple, list)):
        return type(arrays[0])(chain.from_iterable(arrays))
    else:
        raise Exception()


def as_same(array, t, device):
    if issubclass(t, np.ndarray):
        return as_numpy_array(array)
    elif issubclass(t, torch.Tensor):
        return as_tensor(array, device=device)
    elif issubclass(t, list):
        return list(array)
    else:
        raise Exception()


class BatcherPrinter(pprint.PrettyPrinter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dispatch[list.__repr__] = BatcherPrinter.format_array
        self._dispatch[tuple.__repr__] = BatcherPrinter.format_array

    def format_batcher(self, obj, stream, indent, allowance, context, level):
        # Code almost equal to _format_dict, see pprint code
        write = stream.write
        write("Batcher(")
        object_dict = obj.tables
        length = len(object_dict)
        items = [(obj.main_table, obj.tables[obj.main_table]),
                 *list((k, v) for k, v in object_dict.items() if k != obj.main_table)]
        if length:
            # We first try to print inline, and if it is too large then we print it on multiple lines
            self.format_table(items, stream, indent, allowance + 1, context, level, inline=False)
        write('\n' + ' ' * indent + ')')

    def format_array(self, obj, stream, indent, allowance, context, level):
        dtype_str = (
              ("ndarray" if isinstance(obj, np.ndarray) else "tensor" if torch.is_tensor(obj) else str(obj.__class__.__name__)) +
              "[{}]".format(str(obj.dtype) if hasattr(obj, 'dtype') else str(obj.dtypes.values[0]) if hasattr(obj, 'dtypes') and len(set(obj.dtypes.values)) == 1 else 'any')
        )
        stream.write(dtype_str + str(tuple(obj.shape) if hasattr(obj, 'shape') else (len(obj),)))

    def format_columns(self, items, stream, indent, allowance, context, level, inline=False):
        # Code almost equal to _format_dict_items, see pprint code
        indent += self._indent_per_level
        write = stream.write
        last_index = len(items) - 1
        if inline:
            delimnl = ' '
        else:
            delimnl = '\n' + ' ' * indent
            write('\n' + ' ' * indent)
        for i, (key, ent) in enumerate(items):
            last = i == last_index
            write("({})".format(key) + ': ')
            self._format(ent, stream, indent,  # + len(key) + 4,
                         allowance if last else 1,
                         context, level)
            if not last:
                write(delimnl)

    def format_table(self, table, stream, indent, allowance, context, level, inline=False):
        # Code almost equal to _format_dict_table, see pprint code
        indent += self._indent_per_level
        write = stream.write
        last_index = len(table) - 1
        if inline:
            delimnl = ' '
        else:
            delimnl = '\n' + ' ' * indent
            write('\n' + ' ' * indent)

        for i, (key, ent) in enumerate(table):
            last = i == last_index
            write("[{}]".format(key) + ':')
            self.format_columns(ent.data.items(), stream, indent,  # + len(key) + 4,
                                allowance if last else 1,
                                context, level)
            if not last:
                write(delimnl)

    def _repr(self, obj, context, level):
        """Format object for a specific context, returning a string
        and flags indicating whether the representation is 'readable'
        and whether the object represents a recursive construct.
        """
        if isinstance(obj, Batcher) or hasattr(obj, 'shape'):
            return " " * (self._width + 1)
        return super()._repr(obj, context, level)

    def _format(self, obj, stream, indent, allowance, context, level):
        # We dynamically add the types of our namedtuple and namedtuple like
        # classes to the _dispatch object of pprint that maps classes to
        # formatting methods
        # We use a simple criteria (_asdict method) that allows us to use the
        # same formatting on other classes but a more precise one is possible
        if isinstance(obj, Batcher) and type(obj).__repr__ not in self._dispatch:
            self._dispatch[type(obj).__repr__] = BatcherPrinter.format_batcher
        elif hasattr(obj, 'shape') and type(obj).__repr__ not in self._dispatch:
            self._dispatch[type(obj).__repr__] = BatcherPrinter.format_array
        super()._format(obj, stream, indent, allowance, context, level)


class SparseBatchSampler(BatchSampler):
    def __init__(self, batcher, on, batch_size=32, shuffle=False, drop_last=False):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.batcher = batcher
        self.on = on

    def __iter__(self):
        length = len(self.batcher)
        block_begins = np.arange(len(self)) * self.batch_size
        block_ends = np.roll(block_begins, -1)
        block_ends[-1] = block_begins[-1] + self.batch_size
        if self.shuffle:
            init_permut = np.random.permutation(length)
            sorter = np.argsort(
                (getattr(self.batcher[self.on], "getnnz", self.batcher[self.on].sum)(1) + np.random.poisson(1, size=length))[init_permut])
            for i in np.random.permutation(len(block_begins)):
                yield init_permut[sorter[block_begins[i]:block_ends[i]]]
        else:
            sorter = np.argsort(getattr(self.batcher[self.on], "getnnz", self.batcher[self.on].sum)(1))
            for i in range(len(block_begins)):
                yield sorter[block_begins[i]:block_ends[i]]

    def __len__(self):
        if self.drop_last:
            return len(self.batcher) // self.batch_size
        else:
            return (len(self.batcher) + self.batch_size - 1) // self.batch_size


class Table:
    def __init__(self, data, primary_id=None, masks=None, subcolumn_names=None, foreign_ids=None, check=True, batcher=None):
        """

        Parameters
        ----------
        data: dict of (torch.Tensor or numpy.ndarray or scipy.sparse.spmatrix)
        primary_id: str
        masks: dict[str, str]
        subcolumn_names: dict[str, list of str]
        foreign_ids: dict[str, Table]
        check: bool
        """
        self.data = data
        self.primary_id = primary_id
        self.masks = masks or {}
        self.subcolumn_names = subcolumn_names or {}
        self.foreign_ids = foreign_ids or {}
        self.batcher = batcher
        for col_name in foreign_ids:
            mask_name = self.masks.get(col_name, None)
            if mask_name is not None:
                self.masks.setdefault('@'+col_name, mask_name)

    def __iter__(self):
        return iter(self.data.values())

    def __len__(self):
        return next(iter(self)).shape[0]

    @property
    def device(self):
        return getattr(next(iter(self)), 'device', None)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def copy(self):
        table = Table(
            data=dict(self.data),
            masks=dict(self.masks),
            subcolumn_names=dict(self.subcolumn_names),
            foreign_ids=dict(self.foreign_ids),
            primary_id=self.primary_id,
            batcher=self.batcher,
            check=False)
        return table

    @property
    def primary_ids(self):
        """

        Returns
        -------
        torch.Tensor or numpy.ndarray
        """
        return self.data[self.primary_id]

    def compute_foreign_relative_(self, name):
        if '@' + name in self.data:
            return self.data['@' + name]
        referenced_table = self.batcher.tables[self.foreign_ids[name]]

        mask_name = self.masks.get(name, None)
        relative_ids, new_mask = factorize(
            values=self.data[name],
            mask=self.data.get(mask_name, None),
            reference_values=referenced_table.primary_ids,
            freeze_reference=True,
        )[:2]

        self.data['@' + name] = relative_ids
        assert (new_mask is None or new_mask.sum() == np.prod(new_mask.shape)) or not (mask_name is None), f"Unkown ids were found in {name} and no existing mask could be found to mask these values"
        if mask_name is not None and new_mask is not None:
            self.data[mask_name] = new_mask
        return relative_ids

    def compute_foreign_absolute_(self, name):
        if name in self.data:
            return self.data[name]
        referenced_table = self.batcher.tables[self.foreign_ids[name]]
        relative_name = '@' + name

        if issparse(self.data[relative_name]):
            array_to_change = self.data[name] = self.data[relative_name].tocsr(copy=True)
            array_to_change.data = as_numpy_array(referenced_table.primary_ids)[array_to_change.data]
        else:
            self.data[name] = as_same(
                as_array(
                    referenced_table.primary_ids,
                    t=type(referenced_table.primary_ids),
                    device=getattr(referenced_table.primary_ids, 'device', None)
                )[self.data[relative_name]],
                t=type(self.data[relative_name]),
                device=getattr(self.data[relative_name], 'device', None))
        return self.data[name]

    def get_col(self, name):
        if name in self.data:
            return self.data[name]
        else:
            if name.startswith('@'):
                return self.compute_foreign_relative_(name[1:])
            else:
                return self.compute_foreign_absolute_(name)

    def __getitem__(self, key):
        # table["mention_id"]
        if key is None:
            return None
        if isinstance(key, str):
            return self.get_col(key)
        # table["features", ["feat0", "feat1"]]
        elif isinstance(key, tuple):
            (top, *rest) = key
            assert isinstance(top, str)
            data = self.get_col(top)
            if len(key) == 1:
                return data
            if isinstance(rest, str):
                return data[self.subcolumn_names[key].index(rest)]
            elif hasattr(rest, "__iter__"):
                return data[[self.subcolumn_names[key].index(name) for name in rest]]
            else:
                raise Exception()
        # table[["mention_id", ("features", ["feat0", "feat1"]), "label"]]
        elif isinstance(key, list):
            if len(key) > 0 and isinstance(key[0], str):
                new_self = self.copy()
                new_self.data = {
                    col_name: new_self[col_name]
                    for col_name in key
                }
                new_self.foreign_ids = {
                    foreign_id: reference_table
                    for foreign_id, reference_table in self.foreign_ids.items()
                    if foreign_id in key or '@' + foreign_id in key
                }
                new_self.masks = {
                    col_name: mask_name
                    for col_name, mask_name in self.masks.items()
                    if col_name in key
                }
                new_self.subcolumn_names = {
                    col_name: subcolumn_names
                    for col_name, subcolumn_names in self.subcolumn_names.items()
                    if col_name in key
                }
                return new_self
        new_self = self.copy()
        new_self.data = {
            col_name: index_slice(mat, key)
            for col_name, mat in new_self.data.items()
        }
        return new_self

    def __setitem__(self, key, value):
        # TODO checks
        # table["mention_id"] = ...
        if isinstance(key, str):
            self.data[key] = value
        # table["features", ["feat0", "feat1"]] = ...
        elif isinstance(key, tuple):
            if len(key) == 1:
                self.data[key[0]] = value
            else:
                top, rest = key
                assert isinstance(top, str)
                data = self.data[top]
                if isinstance(rest, str):
                    data[self.subcolumn_names[key].index(rest)] = value
                elif hasattr(rest, "__iter__"):
                    data[[self.subcolumn_names[key].index(name) for name in rest]] = value
                else:
                    raise Exception()
        # table[["mention_id", ("features", ["feat0", "feat1"]), "label"]] = ...
        elif isinstance(key, list):
            for part in zip(key, value):
                self[part] = value
        else:
            raise Exception()

    def __delitem__(self, key):
        # TODO checks
        # del table["mention_id"]
        if isinstance(key, str):
            assert not key.startswith('@'), 'You should directly delete column {} instead of {}'.format(key.strip('@'), key)
            del self.data[key]
            if key in self.foreign_ids and '@' + key in self.data:
                del self.data['@' + key]
            if key in self.foreign_ids:
                del self.foreign_ids[key]
            self.masks = {col_name: mask_name for col_name, mask_name in self.masks.items() if mask_name != key}
        # del table[["mention_id", "features", "label"]]
        elif isinstance(key, tuple):
            assert len(key) == 1
            del self.data[key[0]]
        elif isinstance(key, list):
            for part in key:
                del self[part]
        else:
            raise Exception()

    def prune_(self):

        masks_length = {}

        for col_name, mask_name in self.masks.items():
            if col_name not in self.data:
                continue
            if mask_name not in masks_length:
                mask = self.data[mask_name]
                if issparse(mask):
                    if hasattr(mask, 'indices'):
                        if len(mask.indices):
                            max_length = mask.indices.max() + 1
                        else:
                            max_length = 0
                    elif hasattr(mask, 'rows'):
                        max_length = max((max(r, default=-1) + 1 for r in mask.rows), default=0)
                    else:
                        raise Exception(f"Unrecognized mask format for {mask_name}: {mask.__class__}")
                    masks_length[mask_name] = max_length
                    mask.resize(mask.shape[0], masks_length[mask_name])
                else:
                    if 0 in mask.shape:
                        max_length = 0
                    else:
                        max_length = mask.sum(-1).max()
                    masks_length[mask_name] = max_length
                    self.data[mask_name] = mask[:, :masks_length[mask_name]]
            col = self.data[col_name]
            if issparse(col):
                col.resize(col.shape[0], masks_length[mask_name])
            else:
                self.data[col_name] = col[:, :masks_length[mask_name]]

    def densify_(self, device=None, dtypes=None):
        dtypes = dtypes or {}

        new_data = {}
        for col_name, col in self.data.items():
            torch_dtype = dtypes.get(col_name, torch.long if (device is not None and not torch.is_tensor(col) and np.issubdtype(col.dtype, np.integer)) else None)
            col = as_array(col,
                           t=torch.Tensor if (device is not None or col_name in dtypes) else np.ndarray,
                           device=device,
                           dtype=torch_dtype)
            new_data[col_name] = col
        self.data = new_data

    def sparsify_(self, device=None):
        new_data = {}
        densified_masks = {}
        sparsified_masks = {}
        for col_name, mask_name in self.masks.items():
            mask = self.data[mask_name]
            col = self.data[col_name]
            if issparse(mask) and not issparse(col):
                if mask_name not in densified_masks:
                    densified_masks[mask_name] = mask.toarray()
                col = as_numpy_array(col)
                data = col[densified_masks[mask_name]]
                col = mask.copy()
                col.data = data
            elif issparse(col) and not issparse(mask):
                if mask_name not in sparsified_masks:
                    sparsified_masks[mask_name] = mask = csr_matrix(as_numpy_array(mask))
                else:
                    mask = sparsified_masks[mask_name]
            elif not issparse(col) and not issparse(mask):
                data = as_numpy_array(col)[as_numpy_array(mask)]
                if mask_name not in sparsified_masks:
                    sparsified_masks[mask_name] = mask = csr_matrix(as_numpy_array(mask))
                else:
                    mask = sparsified_masks[mask_name]
                col = mask.copy()
                col.data = data
            new_data.setdefault(mask_name, mask.tolil())
            new_data.setdefault(col_name, col.tolil())

        # Then convert the other columns as numpy arrays
        for col_name, col in self.data.items():
            if col_name not in new_data:
                col = as_numpy_array(col)
                new_data[col_name] = col
        self.data = new_data

    def densify(self, device=None, dtypes=None):
        new_self = self.copy()
        new_self.densify_(device, dtypes)
        return new_self

    def sparsify(self):
        new_self = self.copy()
        new_self.sparsify_()
        return new_self

    @property
    def non_relative_data(self):
        keys = list(dict.fromkeys(key[1:] if key.startswith('@') else key for key in self.data.keys()))
        return {name: self[name] for name in keys}

    def fill_absolute_data_(self):
        for name in self.foreign_ids:
            self.compute_foreign_absolute_(name)

    def fill_absolute_data(self):
        new_self = self.copy()
        new_self.fill_absolute_data_()
        return new_self

    @classmethod
    def concat(cls, tables, sparsify=True):
        data = defaultdict(lambda: [])
        for table in tables:
            if sparsify:
                table = table.sparsify()
            for name, col in table.non_relative_data.items():
                data[name].append(col)
        new_data = {name: concat(cols) for name, cols in data.items()}
        new_table = tables[0]
        new_table.data = new_data
        new_table.batcher = None
        return new_table

    def drop_relative_data_(self):
        for name in self.foreign_ids:
            if '@' + name in self.data:
                del self.data['@' + name]

    def __getstate__(self):
        from nlstruct.core.cache import hash_object
        new_self = self.copy()
        for name in self.foreign_ids:
            new_self.compute_foreign_absolute_(name)
        new_self.drop_relative_data_()
        return new_self.__dict__


class Batcher:
    def __init__(self, tables, main_table=None, masks=None, subcolumn_names=None, foreign_ids=None, primary_ids=None, check=True):
        """

        Parameters
        ----------

        tables: dict[str, dict]
        """
        self.main_table = main_table or next(iter(tables.keys()))
        if isinstance(next(iter(tables.values())), Table):
            self.tables = tables
            for table in tables.values():
                table.batcher = self
            return
        subcolumn_names = {table_name: dict(cols) for table_name, cols in subcolumn_names.items()} if subcolumn_names is not None else {}
        if check:
            tables = {table_name: dict(cols) for table_name, cols in tables.items()}
            for table_name, table in tables.items():
                for col_name, col in table.items():
                    if isinstance(col, pd.DataFrame):
                        tables[table_name][col_name] = col.values
                        subcolumn_names.setdefault(table_name, {}).setdefault(col_name, list(col.columns))
                    elif isinstance(col, pd.Series):
                        tables[table_name][col_name] = col.values
                    elif isinstance(col, pd.Categorical):
                        tables[table_name][col_name] = col.codes
                    else:
                        assert col is not None, f"Column {repr(table_name)}{repr(col_name)} cannot be None"
                        tables[table_name][col_name] = col
        if primary_ids is not None:
            primary_ids = primary_ids
        else:
            primary_ids = {table_name: f"{table_name}_id" for table_name in tables if f"{table_name}_id" in tables[table_name]}
            if check:
                for table_name, primary_id in primary_ids.items():
                    uniques, counts = np.unique(tables[table_name][primary_id], return_counts=True)
                    duplicated = uniques[counts > 1]
                    assert len(duplicated) == 0, f"Primary id {repr(primary_id)} of {repr(table_name)} has {len(duplicated)} duplicate{'s' if len(duplicated) > 0 else ''}, " \
                                                 f"when it should be unique: {repr(list(duplicated[:5]))} (first 5 shown)"
        if foreign_ids is None:
            foreign_ids = {}
            for table_name, table_columns in tables.items():
                # ex: table_name = "mention", table_columns = ["sample_id", "begin", "end", "idx_in_sample"]
                for col_name in table_columns:
                    col_name = col_name.strip('@')
                    if col_name.endswith('_id') and col_name != primary_ids.get(table_name, None):
                        prefix = col_name[:-3]
                        foreign_table_name = next((table_name for table_name in tables if prefix.endswith(table_name)), None)
                        # foreign_table_id = f"{table_name}_id"
                        if foreign_table_name is not None:
                            foreign_ids.setdefault(table_name, {})[col_name] = foreign_table_name
        if masks is None:
            masks = {}
            for table_name, table_columns in tables.items():
                # ex: table_name = "mention", table_columns = ["sample_id", "begin", "end", "idx_in_sample"]
                for col_name in table_columns:
                    if col_name.endswith('_mask') and col_name != primary_ids.get(table_name, None):
                        id_name = col_name[:-5] + '_id'
                        # foreign_table_id = f"{table_name}_id"
                        if id_name in table_columns or '@' + id_name in table_columns:
                            masks.setdefault(table_name, {})[id_name] = col_name
        if check:
            # Check that all tables / columns exist in subcolumn names
            for table_name, cols in subcolumn_names.items():
                assert table_name in tables, f"Unknown table {repr(table_name)} in `subcolumn_names`"
                for col_name in cols:
                    assert col_name in tables[table_name], f"Unknown column {repr(col_name)} for table {repr(table_name)} in `subcolumn_names`"
            # Check that all tables / columns exist in masks
            for table_name, cols in masks.items():
                assert table_name in tables, f"Unknown table {repr(table_name)} in `masks`"
                for col_name, mask_name in cols.items():
                    assert col_name in tables[table_name], f"Unknown column {repr(col_name)} for table {repr(table_name)} in `masks`"
                    assert mask_name in tables[table_name], f"Unknown mask {repr(mask_name)} for column {table_name}/{col_name} in `masks`"
            # Check that all tables / columns exist in foreign_ids
            for table_name, cols in foreign_ids.items():
                assert table_name in tables, f"Unknown table {repr(table_name)} in `foreign_ids`"
                for col_name, foreign_table_name in cols.items():
                    assert col_name in tables[table_name], f"Unknown column {repr(col_name)} for table {repr(table_name)} in `foreign_ids`"
                    assert foreign_table_name in tables, f"Unknown foreign table {repr(foreign_table_name)} for column {table_name}/{col_name} in `foreign_ids`"
            # Check that all tables / columns exist in primary_ids
            for table_name, col_name in primary_ids.items():
                assert table_name in tables, f"Unknown table {repr(table_name)} in `primary_ids`"
                assert col_name in tables[table_name], f"Unknown column {repr(col_name)} for table {repr(table_name)} in `primary_ids`"
        self.tables = {
            key: Table(table_data,
                       primary_id=primary_ids.get(key, None),
                       masks=masks.get(key, None),
                       subcolumn_names=subcolumn_names.get(key, None),
                       foreign_ids=foreign_ids.get(key, None),
                       batcher=self)
            for key, table_data in tables.items()
        }  # type: dict[str, Table]

    @property
    def primary_ids(self):
        return self.tables[self.main_table].primary_ids

    @property
    def device(self):
        return getattr(next(iter(self.tables[self.main_table].values())), 'device', None)

    def __len__(self):
        return len(self.tables[self.main_table])

    def keys(self):
        return self.tables[self.main_table].keys()

    def values(self):
        return self.tables[self.main_table].values()

    def items(self):
        return self.tables[self.main_table].items()
    
    def __iter__(self):
        return iter(self.values())

    def __getitem__(self, key):
        table = None
        if isinstance(key, str):
            key = (key,)
        if isinstance(key, tuple):
            if key[0] not in self.tables:
                key = (self.main_table, *key)
            if len(key) == 1:
                self = self.copy()
                self.main_table = key[0]
                return self
            elif isinstance(key[1], str):
                if len(key) == 2:
                    return self.tables[key[0]][key[1]]
                return self.tables[key[0]][key[1:]]
            elif isinstance(key[1], list) and isinstance(key[1][0], str):
                assert len(key) == 2
                return Batcher({key[0]: self.tables[key[0]][key[1]]})
            else:
                assert len(key) == 2
                table, indexer = key
        elif isinstance(key, list):
            if isinstance(key[0], str):
                assert set(type(k) for k in key) == {str}
                return self.slice_tables(key)
        else:
            indexer = key
        
        if isinstance(indexer, slice):
            device = self.device
            if device is None:
                indexer = np.arange(indexer.start or 0, indexer.stop, indexer.step or 1)
            else:
                indexer = torch.arange(indexer.start or 0, indexer.stop, indexer.step or 1, device=device)
        else:
            dtype = getattr(indexer, 'dtype', None)
            if dtype is torch.bool:
                indexer = torch.nonzero(indexer, as_tuple=True)[0]
            elif dtype == np.bool:
                indexer = np.nonzero(indexer)[0]
        return self.query_ids(indexer, table=table)

    def __setitem__(self, key, value):
        if isinstance(key, str):
            assert isinstance(value, Batcher) and len(value.tables.keys()) == 1
            self.tables[key] = next(iter(value.tables.values()))
            self.tables[key].batcher = self
        elif isinstance(key, tuple):
            if len(key) == 2:
                self.tables[key[0]][key[1]] = value
            self.tables[key[0]][key[1:]] = value
        elif isinstance(key, list):
            if isinstance(value, Batcher):
                assert len(value.data) == len(key)
                for name, table in zip(key, value.data.values()):
                    self.tables[name] = table.copy()
                    self.tables[name].batcher = self
            elif not len(key) or isinstance(key[0], (str, tuple, list)):
                for part, val in zip(key, value):
                    self[part] = val
        else:
            raise Exception()

    def __delitem__(self, key):
        if isinstance(key, str):
            self.slice_tables_([name for name in self.tables if name != key])
            return
        elif isinstance(key, tuple):
            if isinstance(key[1], (str, list)):
                assert len(key) == 2
                del self.tables[key[0]][key[1]]
                return
        elif isinstance(key, list):
            if isinstance(key[0], str):
                assert set(type(k) for k in key) == {str}
                self.slice_tables_([name for name in self.tables if name not in key])
                return
        raise Exception()

    def copy(self):
        return Batcher({key: table.copy() for key, table in self.tables.items()}, main_table=self.main_table)

    def query_ids(self, ids, table=None, inplace=False, **densify_kwargs):
        if not inplace:
            self = self.copy()
        if table is None:
            table = self.main_table
        else:
            self.main_table = table
        selected_ids = {table: ids}
        queried_tables = {}
        queue = [table]
        while len(queue):
            table_name = queue.pop(0)
            # Ex: table_name = relations
            # print("query", table_name, len(selected_ids[table_name]), selected_ids[table_name])
            # print("Querying", table_name, len(selected_ids[table_name]))
            table = self.tables[table_name][selected_ids[table_name]]
            table.prune_()
            queried_tables[table_name] = table
            for foreign_id, reference_table in table.foreign_ids.items():
                # print("   Processing foreign", foreign_id, "->", reference_table)
                # Ex: col_name = from_mention_id
                #     foreign_table_name = mention
                #     foreign_table_id = mention_id

                # We don't want to reindex the token_id column in the token table: it's useless and we will
                # moreover need it intact for when we rebuild the original data
                mask_name = table.masks.get(foreign_id, None)
                relative_ids, new_mask, unique_ids = factorize(
                    values=table['@' + foreign_id],
                    mask=table[mask_name],
                    reference_values=selected_ids.get(reference_table, None),
                    # If querying was done against the main axis primary ids (main_table)
                    # then we don't want to any more ids than those that were given
                    # ex: batcher.set_main("relation")[:10] => only returns relations 0, 1, ... 9
                    # If a table refers to other relations through foreign keys, then those pointers will be masked
                    # For non main ids (ex: mentions), we allow different tables to disagree on the mentions to query
                    # and retrieve all of the needed mentions
                    freeze_reference=reference_table in queried_tables,
                )
                # new_col, new_mask, unique_ids = col, queried_table.get(mask_name, None), col.tocsr().data if hasattr(col, 'tocsr') else col#selected_ids.get(foreign_table_name, None)
                if mask_name is not None and new_mask is not None:
                    table[mask_name] = new_mask
                selected_ids[reference_table] = unique_ids
                table['@' + foreign_id] = relative_ids
                if reference_table not in queried_tables:
                    queue.append(reference_table)
                    # print("  Adding table", reference_table, "to queue")

        self.tables.update(queried_tables)
        if len(densify_kwargs):
            self.densify_(**densify_kwargs)
        return self

    def __repr__(self):
        return BatcherPrinter(indent=2, depth=2).pformat(self)

    def densify_(self, device, dtypes=None):
        dtypes = dtypes or {}
        for table in self.tables.values():
            table.densify_(device, dtypes)

    def sparsify_(self):
        for table in self.tables.values():
            table.sparsify_()

    def densify(self, device=None, dtypes=None):
        new_self = self.copy()
        new_self.densify_(device, dtypes)
        return new_self

    def sparsify(self):
        new_self = self.copy()
        new_self.sparsify_()
        return new_self

    def slice_tables_(self, names):
        if self.main_table not in names:
            self.main_table = names[0]
        for name in names:
            table = self.tables[name]
            new_foreign_ids = {}
            for foreign_id, referenced_table_name in table.foreign_ids.items():
                if referenced_table_name not in names:
                    table.compute_foreign_absolute_(foreign_id)
                    if '@' + foreign_id in table.keys():
                        del table.data['@' + foreign_id]
                else:
                    new_foreign_ids[foreign_id] = referenced_table_name
            table.foreign_ids = new_foreign_ids
        for name in list(self.tables.keys()):
            if name not in names:
                del self.tables[name]
        self.tables = {key: self.tables[key] for key in names}

    def slice_tables(self, names):
        new_self = self.copy()
        new_self.slice_tables_(names)
        return new_self

    @classmethod
    def concat(cls, batches, sparsify=True):
        tables = defaultdict(lambda: [])
        for batch in batches:
            for key, table in batch.tables.items():
                tables[key].append(table)
        new_tables = {key: Table.concat(tables, sparsify=sparsify) for key, tables in tables.items()}
        new_batcher = batches[0].copy()
        new_batcher.tables = new_tables
        for table in new_tables.values():
            table.batcher = new_batcher
        return new_batcher

    def drop_duplicates(self, names=None):
        if names is None:
            names = list(self.tables)
        elif isinstance(names, str):
            names = [names]
        self = self.copy()
        for name, table in self.tables.items():
            table.fill_absolute_data_()
            table.drop_relative_data_()
        for name in names:
            table = self.tables[name].fill_absolute_data()
            index = get_deduplicator(table.primary_ids)
            self.tables[name] = table[index]
        return self


    def dataloader(self,
                   batch_size=32,
                   sparse_sort_on=None,
                   shuffle=False,
                   device=None,
                   dtypes=None,
                   **kwargs):
        batch_sampler = kwargs.pop("batch_sampler", None)
        if sparse_sort_on is not None:
            batch_sampler = SparseBatchSampler(self, on=sparse_sort_on, batch_size=batch_size, shuffle=shuffle, drop_last=False)
        else:
            kwargs['batch_size'] = batch_size
            kwargs['shuffle'] = shuffle
        return DataLoader(range(len(self)),  # if self._idx is None else self._idx,
                          collate_fn=lambda ids: self.query_ids(ids, device=device),
                          batch_sampler=batch_sampler,
                          **kwargs)

if __name__ == "__main__":
    batcher = Batcher({
        "doc": {
            "doc_id": np.asarray([10000, 20000, 30000]),
            "token_id": csr_matrix(np.asarray([
                [10001, 10002, 10003, 0],
                [20001, 20002, 20003, 20004],
                [30001, 30002, 0, 0],
            ])),
            "token_mask": csr_matrix(np.asarray([
                [True, True, True, False],
                [True, True, True, True],
                [True, True, False, False],
            ])).astype(bool),
        },
        "token": {
            "token_id": np.asarray([10001, 10002, 10003, 20001, 20002, 20003, 20004, 30001, 30002]),
            "doc_id": np.asarray([10000, 10000, 10000, 20000, 20000, 20000, 20000, 30000, 30000]),
            "word": np.asarray([0, 1, 0, 2, 3, 4, 2, 5, 6]),
        },
    })
    # print(batcher["doc", "@token_id"].toarray())
    print(Batcher.concat([
        batcher["doc", [0, 0, 2]],
        batcher["doc", [1, 0, 2]]
    ])["doc"].densify(torch.device('cpu')).drop_duplicates().sparsify().densify(torch.device('cpu'))["doc", ["doc_id", "token_id"]])
    batcher2 = batcher[[1, 0, 0, 2, 0, 0, 1]]
    print(batcher2["token", "@doc_id"])
    print(batcher2["doc", "doc_id"])
    batcher = batcher["token", [0, 2]]
    print(batcher)
    batcher2 = batcher[["doc"]].densify(torch.device('cpu'))
    print("A", batcher2["doc", "token_id"])
    print(batcher2)
    batcher2 = batcher[["token"]].densify(torch.device('cpu'))
    print("B", batcher2["token", "doc_id"], type(batcher2["token", "doc_id"]))
    print(batcher2)
    print("-----------")
    # batcher["doc", :2]  # get first two docs
    # batcher["doc"].loc[[10000, 30000]]  # get those docs
    # batcher.loc["doc", [10000, 30000]]  # get those docs
    # batcher.to(torch.device('cuda'))  # densify and send to cuda device
    # batcher.sparsify()  # load from any device and sparsify as numpy / csr_matrices
    # batcher["doc"].relative()  # get those docs
    # batcher["doc", "token_id"]  # get the docs token_id
    # batcher["doc", "word"][batcher["doc", "token_id"]]  # get the docs token_id
    # batcher["doc", "word"]  # get the docs words
    # batcher["doc"]["token_id"]  # get those docs
