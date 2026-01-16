import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
import numpy as np

class FeatureExtractorModel:
    def __init__(self, model_id: str, device: str = "cuda", force_seq_len: int = None):
        """
        force_seq_len: DB(hf_snapshots)から取得した強制シーケンス長
        """
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.config = None
        self.hidden_size = 0
        self.required_seq_len = force_seq_len # 引数で指定があればそれを優先
        
        print(f"[Model] Initializing {model_id} on {self.device}...")
        if self.required_seq_len:
            print(f"[Model] Using enforced sequence length from DB: {self.required_seq_len}")

        self._load_model()

    def _load_model(self):
        try:
            self.config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                self.model_id, 
                trust_remote_code=True,
                torch_dtype="auto", 
                device_map=self.device
            )
            self.model.eval()
            
            # Embedding次元数の特定
            if hasattr(self.config, "d_model"):
                self.hidden_size = self.config.d_model
            elif hasattr(self.config, "hidden_size"):
                self.hidden_size = self.config.hidden_size
            elif hasattr(self.config, "n_embd"):
                self.hidden_size = self.config.n_embd
            else:
                self.hidden_size = 768 
            
            # DBからの指定がない場合のみ、Configから読み取る
            if self.required_seq_len is None:
                if hasattr(self.config, "context_length"):
                    self.required_seq_len = self.config.context_length
                    print(f"[Model] Auto-detected context_length: {self.required_seq_len}")
                elif "patchtst" in self.model_id.lower():
                    self.required_seq_len = 512
            
            print(f"[Model] Loaded successfully. Hidden dim: {self.hidden_size}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_id}: {e}")

    def get_embedding_dim(self) -> int:
        return self.hidden_size

    def forward(self, batch_values: torch.Tensor, batch_mask: torch.Tensor) -> torch.Tensor:
        original_len = batch_values.shape[1]
        
        batch_values = batch_values.to(self.device)
        batch_mask = batch_mask.to(self.device)

        # 次元調整 (Batch, Seq) -> (Batch, Seq, 1)
        if batch_values.dim() == 2:
            batch_values = batch_values.unsqueeze(-1)
        if batch_mask.dim() == 2:
            batch_mask = batch_mask.unsqueeze(-1)
        
        # パディング/トリミング
        if self.required_seq_len is not None:
            req_len = int(self.required_seq_len)
            curr_len = batch_values.shape[1]
            
            if curr_len < req_len:
                diff = req_len - curr_len
                batch_values = F.pad(batch_values, (0, 0, diff, 0), value=0.0)
                batch_mask = F.pad(batch_mask, (0, 0, diff, 0), value=0.0)
            elif curr_len > req_len:
                batch_values = batch_values[:, -req_len:, :]
                batch_mask = batch_mask[:, -req_len:, :]

        with torch.no_grad():
            try:
                # PatchTST等は past_values を使用
                outputs = self.model(
                    past_values=batch_values,
                    past_observed_mask=batch_mask,
                    output_hidden_states=True
                )
                
                if hasattr(outputs, "last_hidden_state"):
                    embeddings = outputs.last_hidden_state
                elif hasattr(outputs, "hidden_states"):
                    embeddings = outputs.hidden_states[-1]
                else:
                    raise ValueError("No hidden states found")
                
                # 出力復元
                if self.required_seq_len is not None:
                    out_len = embeddings.shape[1]
                    if original_len < out_len:
                        embeddings = embeddings[:, -original_len:, :]
                        
                return embeddings

            except Exception as e:
                raise RuntimeError(f"Inference failed: {e}")