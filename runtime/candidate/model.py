def _setup_C2():
    SELECTED = 'caimira/native/all'

    import hashlib, json, math, re
    from collections import Counter
    from functools import lru_cache
    from pathlib import Path
    import numpy as np
    from scipy.special import expit, logit

    WIDTH = 384
    BATCH = 1024
    ENTITY_WIDTH = 32

    def validate_texts(texts):
        if not isinstance(texts, (list, tuple)) or any(not isinstance(t, str) for t in texts):
            raise ValueError("features require a sequence of exact original item strings")
        if any(len(t.encode("utf8")) > 64 * 1024**2 for t in texts):
            raise ValueError("item text exceeds the declared64-MiB bound")
        return texts

    def _hash_tokens(tokens):
        counts = [0] * 384
        for token, count in Counter(tokens).items():
            h = hashlib.blake2b(token.encode("utf8"), digest_size=8, person=b"roadmap.lex.v1").digest()
            counts[int.from_bytes(h[:4], "little") % 384] += count if h[4] & 1 else -count
        norm = math.sqrt(math.fsum(x * x for x in counts))
        return [x / norm if norm else 0.0 for x in counts]

    def lexical_features(texts):
        """Hash complete item text only; no subject, benchmark or JSON envelope."""
        validate_texts(texts)
        return np.asarray(
            [_hash_tokens(re.findall(r"\w+|[^\w\s]", t.casefold())) for t in texts], dtype=np.float32
        ).reshape(-1, 384)

    _uncached_lexical = lexical_features

    @lru_cache(maxsize=512)
    def _encode_small(text):
        return _uncached_lexical([text])[0]

    def lexical_features(texts):
        return np.asarray([_encode_small(t) if len(t) <= 32768 else _uncached_lexical([t])[0] for t in texts], dtype=np.float32).reshape(-1, WIDTH)

    class FeatureBank:
        """Cache transform results per distinct string, retaining input alignment."""

        def __init__(self, rows):
            item_texts, subjects, pairs, groups = {}, {}, {}, {}
            indexes = np.empty((len(rows), 4), dtype=np.int64)
            for i, row in enumerate(rows):
                text, subject = row["text"], row["subject"]
                indexes[i] = (
                    item_texts.setdefault(text, len(item_texts)),
                    subjects.setdefault(subject, len(subjects)),
                    pairs.setdefault((row["benchmark"], subject), len(pairs)),
                    groups.setdefault((row["benchmark"], row["group"]), len(groups)),
                )
            self.item, self.subject, self.pair, self.group = indexes.T
            self.x = self._encode(item_texts)
            self.z = self._encode(subjects)

        @staticmethod
        def _encode(strings):
            keys = list(strings)
            matrix = np.empty((len(keys), WIDTH), dtype=np.float32)
            for start in range(0, len(keys), 256):
                matrix[start : start + 256] = lexical_features(keys[start : start + 256])
            return matrix

    def offset(logits, labels):
        """Unique MAP intercept under a unit normal prior. Its root lies in [-31,31]."""
        if not len(labels):
            return 0.0
        low, high = -float(len(labels)), float(len(labels))
        for _ in range(42):
            middle = (low + high) / 2
            if expit(logits + middle).sum() - labels.sum() + middle > 0:
                high = middle
            else:
                low = middle
        return (low + high) / 2

    class Forecaster:
        def __init__(self, name, predictor=None):
            self.name = name
            self.predictor = predictor

        def predict(self, input_pair, labeled):
            subject, item = input_pair
            support = [
                (entry[0][1], entry[1]) for entry in labeled
                if entry[0][0] == subject and entry[0][1]["benchmark_id"] == item["benchmark_id"]
            ]
            if len(support) > 31 or any(type(y) is not int or y not in (0, 1) for _, y in support):
                raise ValueError("invalid purchased prefix")
            y = np.asarray([label for _, label in support], dtype=int)
            if self.name in ("empirical_mean", "pm8"):
                mass = 8 if self.name == "pm8" else 0
                return float((y.sum() + mass / 2) / (len(y) + mass)) if len(y) + mass else 0.5
            s = json.dumps(subject, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            items = [item, *(entry for entry, _ in support)]
            bank = FeatureBank([dict(benchmark=item["benchmark_id"], subject=s, group=str(i), item=str(i),
                                     text=v["item_content"]) for i, v in enumerate(items)])
            query = np.array([0])
            purchased = np.arange(1, len(items))
            adaptation = self.name.split("/")[1]
            if adaptation == "native":
                return float(self.predictor.predict_with_evidence(bank, query, purchased, y)[0])
            p = self.predictor.predict(bank, np.arange(len(items)))
            if adaptation == "fixed" or not len(y):
                return float(p[0])
            logits = logit(np.clip(p, 1e-9, 1 - 1e-9))
            return float(expit(logits[0] + offset(logits[1:], y)))

    from types import SimpleNamespace
    import torch
    from torch import nn
    from scipy.optimize import minimize
    torch.set_num_threads(1)

    def _indices(indices, length):
        a = np.asarray(indices)
        if a.ndim != 1 or (a.size and (a.dtype.kind not in "iu" or a.min() < 0 or a.max() >= length)):
            raise ValueError("invalid feature row indices")
        return a.astype(np.int64, copy=False)

    def _map(offset, loading, mean, labels):
        """Solve the native Gaussian-prior logistic MAP in one or two dimensions."""

        def obj(a):
            eta = offset + loading @ a
            delta = a - mean
            return (np.logaddexp(0, eta) - labels * eta).sum() + 0.5 * (delta @ delta), loading.T @ (
                expit(eta) - labels
            ) + delta

        result = minimize(obj, mean, jac=True, method="L-BFGS-B", options={"gtol": 1e-8, "ftol": 1e-13})
        if not np.isfinite(result.fun) or (not result.success and np.abs(result.jac).max() > 1e-6):
            raise ArithmeticError("structured MAP did not converge")
        return result.x

    SCHEMA = 'proxy-structured-inductive/v1'

    KINDS = ('rasch', 'twopl', 'threepl_global', 'threepl_item', 'mirt', 'caimira')

    FLOOR_CAP = 0.25

    FLOOR_SHRINKAGE = 0.1

    ITEM_FLOOR_SHRINKAGE = 1.0

    CAIMIRA_L1 = 0.0001

    def _torch(device, dtype):
        return torch, getattr(torch, dtype)

    def _array(value, *, ndim, name):
        def boolean(part):
            if isinstance(part, (bool, np.bool_)):
                return True
            if isinstance(part, (list, tuple)):
                return any(boolean(v) for v in part)
            return isinstance(part, np.ndarray) and part.dtype.kind == "b"

        if boolean(value):
            raise ValueError(name + " cannot contain booleans")
        raw = np.asarray(value)
        if raw.dtype.kind not in "fiu" or raw.ndim != ndim or not np.isfinite(raw).all():
            raise ValueError(name + " must be a finite numeric array")
        return np.asarray(raw, dtype=np.float64)

    def _config(kind, rank, prior_kind, dimension, subject_dimension, dtype, ridge, floor_cap):
        if not isinstance(kind, str):
            raise ValueError("kind must be a string")
        if (
            type(dimension) is not int
            or not 1 <= dimension <= 4096
            or type(subject_dimension) is not int
            or not 0 <= subject_dimension <= 4096
        ):
            raise ValueError("invalid item/subject feature dimensions")
        if kind.endswith("_metadata"):
            kind, prior_kind = kind.removesuffix("_metadata"), "metadata"
        kind = {"1pl": "rasch", "2pl": "twopl", "3pl_global": "threepl_global", "3pl_item": "threepl_item"}.get(kind, kind)
        if kind not in KINDS or prior_kind not in ("common", "metadata"):
            raise ValueError("unknown structured recipe/prior")
        if type(rank) is not int or not 1 <= rank <= 8:
            raise ValueError("rank must be an integer in1..8")
        if prior_kind == "metadata" and subject_dimension == 0:
            raise ValueError("metadata prior requires explicit feature columns")
        if dtype not in ("float32", "float64") or type(ridge) not in (float, int) or ridge != 0.01:
            raise ValueError("precision/ridge differs from the fixed recipe")
        if type(floor_cap) not in (int, float) or not math.isfinite(floor_cap) or not 0 <= floor_cap <= 0.25:
            raise ValueError("floor cap must be in[0,.25]")
        return {
            "kind": kind,
            "rank": rank if kind in ("mirt", "caimira") else 1,
            "prior_kind": prior_kind,
            "dimension": dimension,
            "subject_dimension": subject_dimension,
            "dtype": dtype,
            "ridge": 0.01,
            "floor_cap": float(floor_cap),
            "floor_shrinkage": FLOOR_SHRINKAGE,
            "item_floor_shrinkage": ITEM_FLOOR_SHRINKAGE,
            "caimira_l1": CAIMIRA_L1,
            "nuisance_prior": "unit_gaussian_mean_penalty",
            "mirt_gauge": "first_loading_one_bounded_centered_contrasts_training_signed_permutation",
        }

    def _shapes(config, subjects=None):
        d, k, z = config["dimension"], config["rank"], config["subject_dimension"]
        shapes = {"intercept": (1,), "difficulty_weight": (d, k if config["kind"] == "caimira" else 1)}
        if config["kind"] in ("twopl", "threepl_global", "threepl_item"):
            shapes.update(discrimination_weight=(d, 1), discrimination_bias=(1,))
        if config["kind"].startswith("threepl"):
            shapes["floor_raw"] = (1,)
        if config["kind"] == "threepl_item":
            shapes["floor_weight"] = (d, 1)
        if config["kind"] == "mirt":
            shapes.update(contrast_weight=(d, k - 1), contrast_bias=(k - 1,))
        if config["kind"] == "caimira":
            shapes.update(relevance_weight=(d, k), relevance_bias=(k,))
        if config["prior_kind"] == "metadata":
            shapes["prior_weight"] = (z, k)
        if subjects is not None:
            shapes["nuisance"] = (max(0, subjects - 1), k)
        return shapes

    def _tensor(torch, value, device, dtype):
        return torch.as_tensor(np.array(value, copy=True, order="C"), device=device, dtype=dtype)

    def _item_tensor(parameters, X, bank, config, frozen=None):
        """Return additive offset, ability loadings, floors and diagnostic internals."""
        import torch

        kind, k = config["kind"], config["rank"]
        raw_d = X @ parameters["difficulty_weight"]
        mean_d = (bank @ parameters["difficulty_weight"]).mean(0) if frozen is None else frozen["difficulty"]
        difficulty = raw_d - mean_d
        loadings = X.new_ones((len(X), k))
        mean_a, mean_q = X.new_zeros(()), X.new_zeros((max(0, k - 1),))
        if kind in ("twopl", "threepl_global", "threepl_item"):
            log_a = math.log(2) * torch.tanh(X @ parameters["discrimination_weight"] + parameters["discrimination_bias"])
            mean_a = (
                (
                    math.log(2) * torch.tanh(bank @ parameters["discrimination_weight"] + parameters["discrimination_bias"])
                ).mean()
                if frozen is None
                else frozen["discrimination_log"]
            )
            loadings = torch.exp(log_a - mean_a)
        elif kind == "mirt" and k > 1:
            q = torch.tanh(X @ parameters["contrast_weight"] + parameters["contrast_bias"])
            mean_q = (
                torch.tanh(bank @ parameters["contrast_weight"] + parameters["contrast_bias"]).mean(0)
                if frozen is None
                else frozen["contrast"]
            )
            loadings = torch.cat((X.new_ones((len(X), 1)), 0.5 * (q - mean_q)), dim=1)
        elif kind == "caimira":
            scores = X @ parameters["relevance_weight"] + parameters["relevance_bias"]
            loadings = torch.softmax(scores - scores.mean(dim=1, keepdim=True), dim=1)
        offset = parameters["intercept"][0] - (difficulty[:, 0] if kind == "mirt" else (loadings * difficulty).sum(1))
        floors = X.new_zeros((len(X),))
        floor_deviation = X.new_zeros((len(X),))
        if kind.startswith("threepl"):
            if kind == "threepl_item":
                center = bank.mean(0) if frozen is None else frozen["feature"]
                floor_deviation = ((X - center) @ parameters["floor_weight"]).flatten()
            floors = config["floor_cap"] * torch.sigmoid(parameters["floor_raw"][0] + floor_deviation)
        return (
            offset,
            loadings,
            floors,
            {
                "difficulty": difficulty,
                "difficulty_center": mean_d,
                "discrimination_log_center": mean_a,
                "contrast_center": mean_q,
                "floor_deviation": floor_deviation,
            },
        )

    def _prior_tensor(parameters, Z, center, config):
        if config["prior_kind"] == "common":
            return Z.new_zeros((len(Z), config["rank"]))
        present = (Z != 0).any(dim=1, keepdim=True)
        return ((Z - center) @ parameters["prior_weight"]) * present

    class ResidentStructured:
        """Frozen item heads and priors; no training-item or subject lookup path."""

        def __init__(self, state, *, device="cuda", dtype=None):
            if not isinstance(state, dict) or state.get("schema") != SCHEMA:
                raise ValueError("invalid structured state schema")
            if set(state) != {"schema", "config", "parameters", "centers", "orientation"}:
                raise ValueError("unexpected inference state keys, including possible training lookup")
            self.config = dict(state["config"])
            c = self.config
            checked = _config(
                c["kind"],
                c["rank"],
                c["prior_kind"],
                c["dimension"],
                c["subject_dimension"],
                c["dtype"],
                c["ridge"],
                c["floor_cap"],
            )
            if checked != c:
                raise ValueError("structured state recipe differs")
            self.kind, self.rank = c["kind"], c["rank"]
            self.device = device
            self.torch, self.dtype = _torch(device, dtype or c["dtype"])
            shapes = _shapes(c)
            if set(state["parameters"]) != set(shapes):
                raise ValueError("inference parameters contain missing/extra or nuisance terms")
            self.parameters = {}
            for name, shape in shapes.items():
                value = _array(state["parameters"][name], ndim=len(shape), name=name)
                if value.shape != shape:
                    raise ValueError("inference parameter shape differs")
                self.parameters[name] = _tensor(self.torch, value, device, self.dtype)
            center_shapes = {
                "difficulty": (c["rank"] if c["kind"] == "caimira" else 1,),
                "feature": (c["dimension"],),
                "subject": (c["subject_dimension"],),
                "contrast": (max(0, c["rank"] - 1),),
                "discrimination_log": (),
            }
            if set(state["centers"]) != set(center_shapes):
                raise ValueError("frozen center names differ")
            self.centers = {}
            for name, shape in center_shapes.items():
                value = _array(state["centers"][name], ndim=len(shape), name="center " + name)
                if value.shape != shape:
                    raise ValueError("frozen center shape differs")
                self.centers[name] = _tensor(self.torch, value, device, self.dtype)
            orientation = _array(state["orientation"], ndim=2, name="orientation")
            if orientation.shape != (self.rank, self.rank) or not np.allclose(
                orientation.T @ orientation, np.eye(self.rank), atol=1e-12
            ):
                raise ValueError("orientation must be an orthogonal declared coordinate transform")
            if self.kind == "caimira" and (not np.isin(orientation, (0, 1)).all() or not np.all(orientation.sum(0) == 1)):
                raise ValueError("CAIMIRA orientation must preserve simplex coordinates")
            self.orientation = _tensor(self.torch, orientation, device, self.dtype)
            self._state = json.loads(json.dumps(state, allow_nan=False))

        @classmethod
        def from_state(cls, state, *, device="cuda", dtype=None):
            return cls(state, device=device, dtype=dtype)

        def export(self):
            return json.loads(json.dumps(self._state, allow_nan=False))

        def item_parameters_tensor(self, X):
            values = _array(X, ndim=2, name="query item features")
            if values.shape[1] != self.config["dimension"]:
                raise ValueError("unseen item feature dimension differs")
            with self.torch.inference_mode():
                tensor = _tensor(self.torch, values, self.device, self.dtype)
                offset, loading, floor, _ = _item_tensor(self.parameters, tensor, None, self.config, self.centers)
                return offset, loading @ self.orientation, floor

        def item_parameters(self, X):
            return tuple(value.cpu().numpy() for value in self.item_parameters_tensor(X))

        def prior_tensor(self, subject_features):
            values = _array(subject_features, ndim=2, name="query subject metadata")
            if values.shape[1] != self.config["subject_dimension"]:
                raise ValueError("unseen subject feature dimension differs")
            with self.torch.inference_mode():
                return (
                    _prior_tensor(
                        self.parameters,
                        _tensor(self.torch, values, self.device, self.dtype),
                        self.centers["subject"],
                        self.config,
                    )
                    @ self.orientation
                )

        def prior(self, subject_features):
            return self.prior_tensor(subject_features).cpu().numpy()

        def prior_covariance(self):
            return np.eye(self.rank, dtype=np.float64)

    irt = SimpleNamespace(ResidentStructured=ResidentStructured)

    class Predictor:
        def __init__(self, method, state, receipt, device):
            self.method, self.state, self.receipt, self.device = method, state, receipt, device
            self._bank = None
            if method in ("rasch", "caimira"):
                self.model = irt.ResidentStructured(state, device=device)
            elif method in ("ncf", "entitymarformer"):
                self.model = _network(method).to(device).eval()
                self.model.load_state_dict(state, strict=True)
            elif method != "lexical":
                raise ValueError("unknown predictor")


        def predict(self, bank, indices):
            return self.predict_with_evidence(bank, indices, [], [])

        @torch.inference_mode()
        def prepare_prediction(self, bank, indices):
            """Cache frozen item heads and subject priors for supplied query attributes."""
            self._bank = bank
            self._texts, keys = self._lookup(bank.item[indices], len(bank.x))
            self._subjects, subjects = self._lookup(bank.subject[indices], len(bank.z))
            if self.method in ("rasch", "caimira"):
                params = [self.model.item_parameters(bank.x[keys[i : i + 4096]]) for i in range(0, len(keys), 4096)]
                self._offset = np.concatenate([row[0] for row in params])
                self._loading = np.concatenate([row[1] for row in params])
                self._prior = self.model.prior(bank.z[subjects])
            elif self.method == "entitymarformer":

                def projected(matrix, ids, layer):
                    return torch.cat(
                        [
                            layer(torch.as_tensor(matrix[ids[i : i + 4096]], device=self.device))
                            for i in range(0, len(ids), 4096)
                        ]
                    )

                self._item_encoding = projected(bank.x, keys, self.model.item)
                self._subject_encoding = projected(bank.z, subjects, self.model.subject)
                self._entity_context = [
                    (self._key_value(b, self._subject_encoding, 0), self._key_value(b, self._item_encoding, 1))
                    for b in self.model.blocks
                ]

        @staticmethod
        def _lookup(values, size):
            unique = np.unique(values)
            mapping = np.full(size, -1, dtype=np.int64)
            mapping[unique] = np.arange(len(unique))
            return mapping, unique

        @staticmethod
        def _key_value(block, context, relation):
            return tuple(
                (block[name](context) + block["relation_" + name].weight[relation]).reshape(-1, 4, ENTITY_WIDTH // 4)
                for name in ("k", "v")
            )

        def _entity(self, bank, query, support, labels):
            """Project shared purchased evidence once per layer, including both relation shifts."""
            support_ids = self._texts[bank.item[support]]
            support_encoding = self._item_encoding[support_ids] + self.model.observed(
                torch.as_tensor(labels, dtype=torch.int64, device=self.device)
            )
            support_values = [self._key_value(b, support_encoding, 2) for b in self.model.blocks]
            result = []
            for start in range(0, len(query), 4096):
                ix = query[start : start + 4096]
                subject_ids = torch.as_tensor(self._subjects[bank.subject[ix]], device=self.device)
                item_ids = torch.as_tensor(self._texts[bank.item[ix]], device=self.device)
                q = self.model.missing.expand(len(ix), -1)
                for block, ((sk, sv), (ik, iv)), (ek, ev) in zip(
                    self.model.blocks, self._entity_context, support_values, strict=True
                ):
                    h = block["q"](q).reshape(len(ix), 4, -1) / (ENTITY_WIDTH // 4) ** 0.5
                    subject_score = (h * sk[subject_ids]).sum(-1, keepdim=True)
                    item_score = (h * ik[item_ids]).sum(-1, keepdim=True)
                    evidence_score = torch.einsum("bhd,shd->bhs", h, ek)
                    w = torch.softmax(torch.cat((subject_score, item_score, evidence_score), dim=-1), dim=-1)
                    observed = torch.einsum("bhs,shd->bhd", w[:, :, 2:], ev)
                    v = (w[:, :, :1] * sv[subject_ids] + w[:, :, 1:2] * iv[item_ids] + observed).flatten(1)
                    q = block["attention_norm"](q + block["out"](v))
                    q = block["ff_norm"](q + block["ff"](q))
                result.append(self.model.out(q).flatten().sigmoid().cpu().numpy())
            return np.concatenate(result)

        @torch.inference_mode()
        def predict_with_evidence(self, bank, indices, support_indices, support_labels):
            query = _indices(indices, len(bank.item))
            support = _indices(support_indices, len(bank.item))
            labels = np.asarray(support_labels)
            if labels.shape != support.shape or not np.isin(labels, (0, 1)).all() or len(labels) > 31:
                raise ValueError("invalid purchased evidence")
            if not len(query):
                return np.empty(0)
            if len(labels) and len(np.unique(bank.pair[np.r_[query, support]])) != 1:
                raise ValueError("within-pair evidence must match the target pair")
            if self.method in ("rasch", "caimira", "entitymarformer"):
                rows = np.r_[query, support]
                if (
                    self._bank is not bank
                    or np.any(self._texts[bank.item[rows]] < 0)
                    or np.any(self._subjects[bank.subject[rows]] < 0)
                ):
                    self.prepare_prediction(bank, rows)
                if self.method == "entitymarformer":
                    return self._check(self._entity(bank, query, support, labels))
                text = self._texts[bank.item[query]]
                ability = self._prior[self._subjects[bank.subject[query]]]
                if len(labels):
                    bought = self._texts[bank.item[support]]
                    ability = _map(
                        self._offset[bought].astype(np.float64),
                        self._loading[bought].astype(np.float64),
                        ability[0].astype(np.float64),
                        labels,
                    )
                return self._check(expit(self._offset[text] + (self._loading[text] * ability).sum(-1)))
            out = []
            for start in range(0, len(query), BATCH):
                ix = query[start : start + BATCH]
                x, z = bank.x[bank.item[ix]], bank.z[bank.subject[ix]]
                if self.method == "lexical":
                    beta = self.state["beta"]
                    p = expit(beta[0] + (x - self.state["center"]) @ beta[1:])
                else:
                    xb, zb = (torch.as_tensor(t, device=self.device) for t in (x, z))
                    logits = self.model(torch.cat((zb, xb), dim=-1))
                    p = logits.sigmoid().cpu().numpy()
                out.append(p)
            return self._check(np.concatenate(out))

        @staticmethod
        def _check(predictions):
            predictions = predictions.astype(np.float64)
            if not np.isfinite(predictions).all() or np.any((predictions < 0) | (predictions > 1)):
                raise ArithmeticError("invalid forecast")
            return predictions

    STATE = json.loads(Path(__file__).with_name('caimira-requires-state.json').read_text())

    PREDICTOR = Predictor('caimira', STATE, {}, 'cpu')

    FORECASTER = Forecaster(SELECTED, PREDICTOR)

    _predict = FORECASTER.predict

    def predict(input, labeled=None):
        if not input[1].get('benchmark_id'):
            return _predict([input[0], dict(input[1], benchmark_id='')], [])
        return _predict(input, [e for e in (labeled or []) if e[0][1].get('benchmark_id')])

    return predict

_C2 = _setup_C2()

def _setup_R():
    SELECTED = 'rasch/offset/all'

    import hashlib, json, math, re
    from collections import Counter
    from functools import lru_cache
    from pathlib import Path
    import numpy as np
    from scipy.special import expit, logit

    WIDTH = 384
    BATCH = 1024
    ENTITY_WIDTH = 32

    def validate_texts(texts):
        if not isinstance(texts, (list, tuple)) or any(not isinstance(t, str) for t in texts):
            raise ValueError("features require a sequence of exact original item strings")
        if any(len(t.encode("utf8")) > 64 * 1024**2 for t in texts):
            raise ValueError("item text exceeds the declared64-MiB bound")
        return texts

    def _hash_tokens(tokens):
        counts = [0] * 384
        for token, count in Counter(tokens).items():
            h = hashlib.blake2b(token.encode("utf8"), digest_size=8, person=b"roadmap.lex.v1").digest()
            counts[int.from_bytes(h[:4], "little") % 384] += count if h[4] & 1 else -count
        norm = math.sqrt(math.fsum(x * x for x in counts))
        return [x / norm if norm else 0.0 for x in counts]

    def lexical_features(texts):
        """Hash complete item text only; no subject, benchmark or JSON envelope."""
        validate_texts(texts)
        return np.asarray(
            [_hash_tokens(re.findall(r"\w+|[^\w\s]", t.casefold())) for t in texts], dtype=np.float32
        ).reshape(-1, 384)

    _uncached_lexical = lexical_features

    @lru_cache(maxsize=512)
    def _encode_small(text):
        return _uncached_lexical([text])[0]

    def lexical_features(texts):
        return np.asarray([_encode_small(t) if len(t) <= 32768 else _uncached_lexical([t])[0] for t in texts], dtype=np.float32).reshape(-1, WIDTH)

    class FeatureBank:
        """Cache transform results per distinct string, retaining input alignment."""

        def __init__(self, rows):
            item_texts, subjects, pairs, groups = {}, {}, {}, {}
            indexes = np.empty((len(rows), 4), dtype=np.int64)
            for i, row in enumerate(rows):
                text, subject = row["text"], row["subject"]
                indexes[i] = (
                    item_texts.setdefault(text, len(item_texts)),
                    subjects.setdefault(subject, len(subjects)),
                    pairs.setdefault((row["benchmark"], subject), len(pairs)),
                    groups.setdefault((row["benchmark"], row["group"]), len(groups)),
                )
            self.item, self.subject, self.pair, self.group = indexes.T
            self.x = self._encode(item_texts)
            self.z = self._encode(subjects)

        @staticmethod
        def _encode(strings):
            keys = list(strings)
            matrix = np.empty((len(keys), WIDTH), dtype=np.float32)
            for start in range(0, len(keys), 256):
                matrix[start : start + 256] = lexical_features(keys[start : start + 256])
            return matrix

    def offset(logits, labels):
        """Unique MAP intercept under a unit normal prior. Its root lies in [-31,31]."""
        if not len(labels):
            return 0.0
        low, high = -float(len(labels)), float(len(labels))
        for _ in range(42):
            middle = (low + high) / 2
            if expit(logits + middle).sum() - labels.sum() + middle > 0:
                high = middle
            else:
                low = middle
        return (low + high) / 2

    class Forecaster:
        def __init__(self, name, predictor=None):
            self.name = name
            self.predictor = predictor

        def predict(self, input_pair, labeled):
            subject, item = input_pair
            support = [
                (entry[0][1], entry[1]) for entry in labeled
                if entry[0][0] == subject and entry[0][1]["benchmark_id"] == item["benchmark_id"]
            ]
            if len(support) > 31 or any(type(y) is not int or y not in (0, 1) for _, y in support):
                raise ValueError("invalid purchased prefix")
            y = np.asarray([label for _, label in support], dtype=int)
            if self.name in ("empirical_mean", "pm8"):
                mass = 8 if self.name == "pm8" else 0
                return float((y.sum() + mass / 2) / (len(y) + mass)) if len(y) + mass else 0.5
            s = json.dumps(subject, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            items = [item, *(entry for entry, _ in support)]
            bank = FeatureBank([dict(benchmark=item["benchmark_id"], subject=s, group=str(i), item=str(i),
                                     text=v["item_content"]) for i, v in enumerate(items)])
            query = np.array([0])
            purchased = np.arange(1, len(items))
            adaptation = self.name.split("/")[1]
            if adaptation == "native":
                return float(self.predictor.predict_with_evidence(bank, query, purchased, y)[0])
            p = self.predictor.predict(bank, np.arange(len(items)))
            if adaptation == "fixed" or not len(y):
                return float(p[0])
            logits = logit(np.clip(p, 1e-9, 1 - 1e-9))
            return float(expit(logits[0] + offset(logits[1:], y)))

    from types import SimpleNamespace
    import torch
    from torch import nn
    from scipy.optimize import minimize
    torch.set_num_threads(1)

    def _indices(indices, length):
        a = np.asarray(indices)
        if a.ndim != 1 or (a.size and (a.dtype.kind not in "iu" or a.min() < 0 or a.max() >= length)):
            raise ValueError("invalid feature row indices")
        return a.astype(np.int64, copy=False)

    def _map(offset, loading, mean, labels):
        """Solve the native Gaussian-prior logistic MAP in one or two dimensions."""

        def obj(a):
            eta = offset + loading @ a
            delta = a - mean
            return (np.logaddexp(0, eta) - labels * eta).sum() + 0.5 * (delta @ delta), loading.T @ (
                expit(eta) - labels
            ) + delta

        result = minimize(obj, mean, jac=True, method="L-BFGS-B", options={"gtol": 1e-8, "ftol": 1e-13})
        if not np.isfinite(result.fun) or (not result.success and np.abs(result.jac).max() > 1e-6):
            raise ArithmeticError("structured MAP did not converge")
        return result.x

    SCHEMA = 'proxy-structured-inductive/v1'

    KINDS = ('rasch', 'twopl', 'threepl_global', 'threepl_item', 'mirt', 'caimira')

    FLOOR_CAP = 0.25

    FLOOR_SHRINKAGE = 0.1

    ITEM_FLOOR_SHRINKAGE = 1.0

    CAIMIRA_L1 = 0.0001

    def _torch(device, dtype):
        return torch, getattr(torch, dtype)

    def _array(value, *, ndim, name):
        def boolean(part):
            if isinstance(part, (bool, np.bool_)):
                return True
            if isinstance(part, (list, tuple)):
                return any(boolean(v) for v in part)
            return isinstance(part, np.ndarray) and part.dtype.kind == "b"

        if boolean(value):
            raise ValueError(name + " cannot contain booleans")
        raw = np.asarray(value)
        if raw.dtype.kind not in "fiu" or raw.ndim != ndim or not np.isfinite(raw).all():
            raise ValueError(name + " must be a finite numeric array")
        return np.asarray(raw, dtype=np.float64)

    def _config(kind, rank, prior_kind, dimension, subject_dimension, dtype, ridge, floor_cap):
        if not isinstance(kind, str):
            raise ValueError("kind must be a string")
        if (
            type(dimension) is not int
            or not 1 <= dimension <= 4096
            or type(subject_dimension) is not int
            or not 0 <= subject_dimension <= 4096
        ):
            raise ValueError("invalid item/subject feature dimensions")
        if kind.endswith("_metadata"):
            kind, prior_kind = kind.removesuffix("_metadata"), "metadata"
        kind = {"1pl": "rasch", "2pl": "twopl", "3pl_global": "threepl_global", "3pl_item": "threepl_item"}.get(kind, kind)
        if kind not in KINDS or prior_kind not in ("common", "metadata"):
            raise ValueError("unknown structured recipe/prior")
        if type(rank) is not int or not 1 <= rank <= 8:
            raise ValueError("rank must be an integer in1..8")
        if prior_kind == "metadata" and subject_dimension == 0:
            raise ValueError("metadata prior requires explicit feature columns")
        if dtype not in ("float32", "float64") or type(ridge) not in (float, int) or ridge != 0.01:
            raise ValueError("precision/ridge differs from the fixed recipe")
        if type(floor_cap) not in (int, float) or not math.isfinite(floor_cap) or not 0 <= floor_cap <= 0.25:
            raise ValueError("floor cap must be in[0,.25]")
        return {
            "kind": kind,
            "rank": rank if kind in ("mirt", "caimira") else 1,
            "prior_kind": prior_kind,
            "dimension": dimension,
            "subject_dimension": subject_dimension,
            "dtype": dtype,
            "ridge": 0.01,
            "floor_cap": float(floor_cap),
            "floor_shrinkage": FLOOR_SHRINKAGE,
            "item_floor_shrinkage": ITEM_FLOOR_SHRINKAGE,
            "caimira_l1": CAIMIRA_L1,
            "nuisance_prior": "unit_gaussian_mean_penalty",
            "mirt_gauge": "first_loading_one_bounded_centered_contrasts_training_signed_permutation",
        }

    def _shapes(config, subjects=None):
        d, k, z = config["dimension"], config["rank"], config["subject_dimension"]
        shapes = {"intercept": (1,), "difficulty_weight": (d, k if config["kind"] == "caimira" else 1)}
        if config["kind"] in ("twopl", "threepl_global", "threepl_item"):
            shapes.update(discrimination_weight=(d, 1), discrimination_bias=(1,))
        if config["kind"].startswith("threepl"):
            shapes["floor_raw"] = (1,)
        if config["kind"] == "threepl_item":
            shapes["floor_weight"] = (d, 1)
        if config["kind"] == "mirt":
            shapes.update(contrast_weight=(d, k - 1), contrast_bias=(k - 1,))
        if config["kind"] == "caimira":
            shapes.update(relevance_weight=(d, k), relevance_bias=(k,))
        if config["prior_kind"] == "metadata":
            shapes["prior_weight"] = (z, k)
        if subjects is not None:
            shapes["nuisance"] = (max(0, subjects - 1), k)
        return shapes

    def _tensor(torch, value, device, dtype):
        return torch.as_tensor(np.array(value, copy=True, order="C"), device=device, dtype=dtype)

    def _item_tensor(parameters, X, bank, config, frozen=None):
        """Return additive offset, ability loadings, floors and diagnostic internals."""
        import torch

        kind, k = config["kind"], config["rank"]
        raw_d = X @ parameters["difficulty_weight"]
        mean_d = (bank @ parameters["difficulty_weight"]).mean(0) if frozen is None else frozen["difficulty"]
        difficulty = raw_d - mean_d
        loadings = X.new_ones((len(X), k))
        mean_a, mean_q = X.new_zeros(()), X.new_zeros((max(0, k - 1),))
        if kind in ("twopl", "threepl_global", "threepl_item"):
            log_a = math.log(2) * torch.tanh(X @ parameters["discrimination_weight"] + parameters["discrimination_bias"])
            mean_a = (
                (
                    math.log(2) * torch.tanh(bank @ parameters["discrimination_weight"] + parameters["discrimination_bias"])
                ).mean()
                if frozen is None
                else frozen["discrimination_log"]
            )
            loadings = torch.exp(log_a - mean_a)
        elif kind == "mirt" and k > 1:
            q = torch.tanh(X @ parameters["contrast_weight"] + parameters["contrast_bias"])
            mean_q = (
                torch.tanh(bank @ parameters["contrast_weight"] + parameters["contrast_bias"]).mean(0)
                if frozen is None
                else frozen["contrast"]
            )
            loadings = torch.cat((X.new_ones((len(X), 1)), 0.5 * (q - mean_q)), dim=1)
        elif kind == "caimira":
            scores = X @ parameters["relevance_weight"] + parameters["relevance_bias"]
            loadings = torch.softmax(scores - scores.mean(dim=1, keepdim=True), dim=1)
        offset = parameters["intercept"][0] - (difficulty[:, 0] if kind == "mirt" else (loadings * difficulty).sum(1))
        floors = X.new_zeros((len(X),))
        floor_deviation = X.new_zeros((len(X),))
        if kind.startswith("threepl"):
            if kind == "threepl_item":
                center = bank.mean(0) if frozen is None else frozen["feature"]
                floor_deviation = ((X - center) @ parameters["floor_weight"]).flatten()
            floors = config["floor_cap"] * torch.sigmoid(parameters["floor_raw"][0] + floor_deviation)
        return (
            offset,
            loadings,
            floors,
            {
                "difficulty": difficulty,
                "difficulty_center": mean_d,
                "discrimination_log_center": mean_a,
                "contrast_center": mean_q,
                "floor_deviation": floor_deviation,
            },
        )

    def _prior_tensor(parameters, Z, center, config):
        if config["prior_kind"] == "common":
            return Z.new_zeros((len(Z), config["rank"]))
        present = (Z != 0).any(dim=1, keepdim=True)
        return ((Z - center) @ parameters["prior_weight"]) * present

    class ResidentStructured:
        """Frozen item heads and priors; no training-item or subject lookup path."""

        def __init__(self, state, *, device="cuda", dtype=None):
            if not isinstance(state, dict) or state.get("schema") != SCHEMA:
                raise ValueError("invalid structured state schema")
            if set(state) != {"schema", "config", "parameters", "centers", "orientation"}:
                raise ValueError("unexpected inference state keys, including possible training lookup")
            self.config = dict(state["config"])
            c = self.config
            checked = _config(
                c["kind"],
                c["rank"],
                c["prior_kind"],
                c["dimension"],
                c["subject_dimension"],
                c["dtype"],
                c["ridge"],
                c["floor_cap"],
            )
            if checked != c:
                raise ValueError("structured state recipe differs")
            self.kind, self.rank = c["kind"], c["rank"]
            self.device = device
            self.torch, self.dtype = _torch(device, dtype or c["dtype"])
            shapes = _shapes(c)
            if set(state["parameters"]) != set(shapes):
                raise ValueError("inference parameters contain missing/extra or nuisance terms")
            self.parameters = {}
            for name, shape in shapes.items():
                value = _array(state["parameters"][name], ndim=len(shape), name=name)
                if value.shape != shape:
                    raise ValueError("inference parameter shape differs")
                self.parameters[name] = _tensor(self.torch, value, device, self.dtype)
            center_shapes = {
                "difficulty": (c["rank"] if c["kind"] == "caimira" else 1,),
                "feature": (c["dimension"],),
                "subject": (c["subject_dimension"],),
                "contrast": (max(0, c["rank"] - 1),),
                "discrimination_log": (),
            }
            if set(state["centers"]) != set(center_shapes):
                raise ValueError("frozen center names differ")
            self.centers = {}
            for name, shape in center_shapes.items():
                value = _array(state["centers"][name], ndim=len(shape), name="center " + name)
                if value.shape != shape:
                    raise ValueError("frozen center shape differs")
                self.centers[name] = _tensor(self.torch, value, device, self.dtype)
            orientation = _array(state["orientation"], ndim=2, name="orientation")
            if orientation.shape != (self.rank, self.rank) or not np.allclose(
                orientation.T @ orientation, np.eye(self.rank), atol=1e-12
            ):
                raise ValueError("orientation must be an orthogonal declared coordinate transform")
            if self.kind == "caimira" and (not np.isin(orientation, (0, 1)).all() or not np.all(orientation.sum(0) == 1)):
                raise ValueError("CAIMIRA orientation must preserve simplex coordinates")
            self.orientation = _tensor(self.torch, orientation, device, self.dtype)
            self._state = json.loads(json.dumps(state, allow_nan=False))

        @classmethod
        def from_state(cls, state, *, device="cuda", dtype=None):
            return cls(state, device=device, dtype=dtype)

        def export(self):
            return json.loads(json.dumps(self._state, allow_nan=False))

        def item_parameters_tensor(self, X):
            values = _array(X, ndim=2, name="query item features")
            if values.shape[1] != self.config["dimension"]:
                raise ValueError("unseen item feature dimension differs")
            with self.torch.inference_mode():
                tensor = _tensor(self.torch, values, self.device, self.dtype)
                offset, loading, floor, _ = _item_tensor(self.parameters, tensor, None, self.config, self.centers)
                return offset, loading @ self.orientation, floor

        def item_parameters(self, X):
            return tuple(value.cpu().numpy() for value in self.item_parameters_tensor(X))

        def prior_tensor(self, subject_features):
            values = _array(subject_features, ndim=2, name="query subject metadata")
            if values.shape[1] != self.config["subject_dimension"]:
                raise ValueError("unseen subject feature dimension differs")
            with self.torch.inference_mode():
                return (
                    _prior_tensor(
                        self.parameters,
                        _tensor(self.torch, values, self.device, self.dtype),
                        self.centers["subject"],
                        self.config,
                    )
                    @ self.orientation
                )

        def prior(self, subject_features):
            return self.prior_tensor(subject_features).cpu().numpy()

        def prior_covariance(self):
            return np.eye(self.rank, dtype=np.float64)

    irt = SimpleNamespace(ResidentStructured=ResidentStructured)

    class Predictor:
        def __init__(self, method, state, receipt, device):
            self.method, self.state, self.receipt, self.device = method, state, receipt, device
            self._bank = None
            if method in ("rasch", "caimira"):
                self.model = irt.ResidentStructured(state, device=device)
            elif method in ("ncf", "entitymarformer"):
                self.model = _network(method).to(device).eval()
                self.model.load_state_dict(state, strict=True)
            elif method != "lexical":
                raise ValueError("unknown predictor")


        def predict(self, bank, indices):
            return self.predict_with_evidence(bank, indices, [], [])

        @torch.inference_mode()
        def prepare_prediction(self, bank, indices):
            """Cache frozen item heads and subject priors for supplied query attributes."""
            self._bank = bank
            self._texts, keys = self._lookup(bank.item[indices], len(bank.x))
            self._subjects, subjects = self._lookup(bank.subject[indices], len(bank.z))
            if self.method in ("rasch", "caimira"):
                params = [self.model.item_parameters(bank.x[keys[i : i + 4096]]) for i in range(0, len(keys), 4096)]
                self._offset = np.concatenate([row[0] for row in params])
                self._loading = np.concatenate([row[1] for row in params])
                self._prior = self.model.prior(bank.z[subjects])
            elif self.method == "entitymarformer":

                def projected(matrix, ids, layer):
                    return torch.cat(
                        [
                            layer(torch.as_tensor(matrix[ids[i : i + 4096]], device=self.device))
                            for i in range(0, len(ids), 4096)
                        ]
                    )

                self._item_encoding = projected(bank.x, keys, self.model.item)
                self._subject_encoding = projected(bank.z, subjects, self.model.subject)
                self._entity_context = [
                    (self._key_value(b, self._subject_encoding, 0), self._key_value(b, self._item_encoding, 1))
                    for b in self.model.blocks
                ]

        @staticmethod
        def _lookup(values, size):
            unique = np.unique(values)
            mapping = np.full(size, -1, dtype=np.int64)
            mapping[unique] = np.arange(len(unique))
            return mapping, unique

        @staticmethod
        def _key_value(block, context, relation):
            return tuple(
                (block[name](context) + block["relation_" + name].weight[relation]).reshape(-1, 4, ENTITY_WIDTH // 4)
                for name in ("k", "v")
            )

        def _entity(self, bank, query, support, labels):
            """Project shared purchased evidence once per layer, including both relation shifts."""
            support_ids = self._texts[bank.item[support]]
            support_encoding = self._item_encoding[support_ids] + self.model.observed(
                torch.as_tensor(labels, dtype=torch.int64, device=self.device)
            )
            support_values = [self._key_value(b, support_encoding, 2) for b in self.model.blocks]
            result = []
            for start in range(0, len(query), 4096):
                ix = query[start : start + 4096]
                subject_ids = torch.as_tensor(self._subjects[bank.subject[ix]], device=self.device)
                item_ids = torch.as_tensor(self._texts[bank.item[ix]], device=self.device)
                q = self.model.missing.expand(len(ix), -1)
                for block, ((sk, sv), (ik, iv)), (ek, ev) in zip(
                    self.model.blocks, self._entity_context, support_values, strict=True
                ):
                    h = block["q"](q).reshape(len(ix), 4, -1) / (ENTITY_WIDTH // 4) ** 0.5
                    subject_score = (h * sk[subject_ids]).sum(-1, keepdim=True)
                    item_score = (h * ik[item_ids]).sum(-1, keepdim=True)
                    evidence_score = torch.einsum("bhd,shd->bhs", h, ek)
                    w = torch.softmax(torch.cat((subject_score, item_score, evidence_score), dim=-1), dim=-1)
                    observed = torch.einsum("bhs,shd->bhd", w[:, :, 2:], ev)
                    v = (w[:, :, :1] * sv[subject_ids] + w[:, :, 1:2] * iv[item_ids] + observed).flatten(1)
                    q = block["attention_norm"](q + block["out"](v))
                    q = block["ff_norm"](q + block["ff"](q))
                result.append(self.model.out(q).flatten().sigmoid().cpu().numpy())
            return np.concatenate(result)

        @torch.inference_mode()
        def predict_with_evidence(self, bank, indices, support_indices, support_labels):
            query = _indices(indices, len(bank.item))
            support = _indices(support_indices, len(bank.item))
            labels = np.asarray(support_labels)
            if labels.shape != support.shape or not np.isin(labels, (0, 1)).all() or len(labels) > 31:
                raise ValueError("invalid purchased evidence")
            if not len(query):
                return np.empty(0)
            if len(labels) and len(np.unique(bank.pair[np.r_[query, support]])) != 1:
                raise ValueError("within-pair evidence must match the target pair")
            if self.method in ("rasch", "caimira", "entitymarformer"):
                rows = np.r_[query, support]
                if (
                    self._bank is not bank
                    or np.any(self._texts[bank.item[rows]] < 0)
                    or np.any(self._subjects[bank.subject[rows]] < 0)
                ):
                    self.prepare_prediction(bank, rows)
                if self.method == "entitymarformer":
                    return self._check(self._entity(bank, query, support, labels))
                text = self._texts[bank.item[query]]
                ability = self._prior[self._subjects[bank.subject[query]]]
                if len(labels):
                    bought = self._texts[bank.item[support]]
                    ability = _map(
                        self._offset[bought].astype(np.float64),
                        self._loading[bought].astype(np.float64),
                        ability[0].astype(np.float64),
                        labels,
                    )
                return self._check(expit(self._offset[text] + (self._loading[text] * ability).sum(-1)))
            out = []
            for start in range(0, len(query), BATCH):
                ix = query[start : start + BATCH]
                x, z = bank.x[bank.item[ix]], bank.z[bank.subject[ix]]
                if self.method == "lexical":
                    beta = self.state["beta"]
                    p = expit(beta[0] + (x - self.state["center"]) @ beta[1:])
                else:
                    xb, zb = (torch.as_tensor(t, device=self.device) for t in (x, z))
                    logits = self.model(torch.cat((zb, xb), dim=-1))
                    p = logits.sigmoid().cpu().numpy()
                out.append(p)
            return self._check(np.concatenate(out))

        @staticmethod
        def _check(predictions):
            predictions = predictions.astype(np.float64)
            if not np.isfinite(predictions).all() or np.any((predictions < 0) | (predictions > 1)):
                raise ArithmeticError("invalid forecast")
            return predictions

    STATE = json.loads(Path(__file__).with_name('rasch-requires-state.json').read_text())

    PREDICTOR = Predictor('rasch', STATE, {}, 'cpu')

    FORECASTER = Forecaster(SELECTED, PREDICTOR)

    _predict = FORECASTER.predict

    def predict(input, labeled=None):
        if not input[1].get('benchmark_id'):
            return _predict([input[0], dict(input[1], benchmark_id='')], [])
        return _predict(input, [e for e in (labeled or []) if e[0][1].get('benchmark_id')])

    return predict

_R = _setup_R()


def _mean(success, n, mass):
    return (success + 0.5 * mass) / (n + mass)

def predict(input, labeled=None):
    benchmark = input[1].get('benchmark_id')
    known = [e for e in (labeled or []) if benchmark and e[0][0] == input[0] and e[0][1].get('benchmark_id') == benchmark]
    n = len(known)
    success = sum(e[1] for e in known)
    p = (0.5 * _C2(input, known)) + (0.5 * _R(input, known))
    return float(p)
