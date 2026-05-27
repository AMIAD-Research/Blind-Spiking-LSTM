from pathlib import Path
from flax import nnx
import orbax.checkpoint as ocp
import jax.numpy as jnp
import jax
from functools import partial
import torch
import numpy as np
from optax.losses import sigmoid_binary_cross_entropy


def save_model(path, model):
	ckpt_dir = Path(path).absolute()
	ckpt_dir.mkdir(parents=True, exist_ok=True)
	_, state = nnx.split(model)
	ckptr = ocp.PyTreeCheckpointer()
	save_args = ocp.args.StandardSave(item=state)
	ckptr.save(str(ckpt_dir), nnx.state(model), force=True)
	return None

def process_raw_dict(raw_state_dict):
  flattened = nnx.traversals.flatten_mapping(raw_state_dict)
  # Cut the '.value' postfix on every leaf path.
  flattened = {(path[:-1] if path[-1] == 'value' else path): value
               for path, value in flattened.items()}
  return nnx.traversals.unflatten_mapping(flattened)

def load_model(path, model):
  """Function to load parameters to a model

  Args:
      path (str): path to the parameters to load
      model (nnx.module): Model

  Returns:
      _type_: _description_
  """
  ckpt_dir = Path(path).absolute()
  ckptr = ocp.StandardCheckpointer()
  raw_dict = ckptr.restore(ckpt_dir)
  abs_model = nnx.eval_shape(lambda:model)
  graph_def, state = nnx.split(abs_model)
  nnx.replace_by_pure_dict(state, process_raw_dict(raw_dict))
  restored_model = nnx.merge(graph_def, state)
  return restored_model

def get_logits(y_hat):
  return jnp.where(nnx.sigmoid(y_hat)>0.5,1,0)

def get_clip(y_hat):
   return y_hat

def loss_fn_single_sentence(model, batch):
  """Loss function for the task Single Sentence

  Args:
      model (_type_): _description_
      batch (_type_): _description_

  Returns:
      _type_: _description_
  """
  x,seq_len,y = batch
  y_hat = model(x)
  #breakpoint()
  y_hat = y_hat[jnp.arange(y_hat.shape[0]),seq_len-1].flatten()
  #breakpoint()

  return sigmoid_binary_cross_entropy(y_hat,y).mean(), get_logits(y_hat) 
          
def loss_fn_sim_par(model, batch):
  """Loss function for the task Similarity and Paraphrase

  Args:
      model (_type_): _description_
      batch (_type_): _description_

  Returns:
      _type_: _description_
  """
  x,seq_len,y = batch
  batch_size = x.shape[0]//2
  h_hat = model(x)
  h_hat = h_hat[jnp.arange(h_hat.shape[0]),seq_len-1]
  first_sentence = h_hat[:batch_size]
  second_sentence = h_hat[batch_size:]
  h = jnp.concat([first_sentence,second_sentence],axis=-1)
  y_hat = model.head(h).flatten()
  return sigmoid_binary_cross_entropy(y_hat,y).mean(), get_logits(y_hat)





@jax.jit
def f1_score(logits, labels):
  """
  Calcule le F1-score pour une classification binaire.
  
  Args:
      logits: Les sorties brutes du modèle (avant sigmoid).
      labels: Les étiquettes réelles (0 ou 1).
      threshold: Le seuil de décision pour la classification.
  """
  # Conversion des logits en prédictions binaire
  logits = logits.astype(jnp.bool_)
  labels = labels.astype(jnp.bool_)
  
  tp = jnp.sum(jnp.logical_and(logits, labels))
  fp = jnp.sum(jnp.logical_and(logits, jnp.logical_not(labels)))
  fn = jnp.sum(jnp.logical_and(jnp.logical_not(logits), labels))
  
  precision = tp / (tp + fp + 1e-7)
  recall = tp / (tp + fn + 1e-7)
  
  f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
  return f1

@jax.jit
def mcc(logits, labels):
  """
  Calcule le coefficient de corrélation de Matthews (MCC).
  
  Args:
      logits: Sorties brutes du modèle (avant activation).
      labels: Étiquettes réelles (0 ou 1).
      threshold: Seuil de classification.
  """
  preds = logits.astype(jnp.bool_)
  labels = labels.astype(jnp.bool_)
  
  # Calcul des quatre composantes de la matrice de confusion
  tp = jnp.sum(jnp.logical_and(preds == True,  labels == True))
  tn = jnp.sum(jnp.logical_and(preds == False, labels == False))
  fp = jnp.sum(jnp.logical_and(preds == True,  labels == False))
  fn = jnp.sum(jnp.logical_and(preds == False, labels == True))
  
  # Calcul du numérateur
  numerator = (tp * tn) - (fp * fn)
  
  # Calcul du dénominateur avec précaution pour éviter la division par zéro
  # On utilise jnp.prod pour plus de clarté dans l'expression
  denominator = jnp.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  
  # JAX gère les NaNs, mais on assure la stabilité
  return jnp.where(denominator == 0, 0.0, numerator / denominator).mean()

@jax.jit
def accuracy(logits,labels):
  return jnp.mean(logits==labels).mean()





@partial(nnx.jit,static_argnames=["loss_fn"])
def train_step(model, optimizer: nnx.Optimizer,loss_fn, batch):
  """Train for a single step."""
  grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
  (loss, y_hat), grads = grad_fn(model, batch)
  # eta_grad = grads["lstm"]["eta"].value
  # jax.debug.print("Graident eta : {x}",x=jnp.abs(eta_grad).mean())
  # f_grad = grads["lstm"]["linearf"]["linear"]["kernel"].value
  # jax.debug.print("Graident f : {x}",x=jnp.abs(f_grad).mean())
  optimizer.update(model,grads)  # In-place updates.
  return loss,y_hat

def process_batch(batch, task):
  """This function processes the batch according to the task

  Args:
      batch (dict): _description_
      task (str): _description_

  Returns:
      Tuple: processed batched
  """
  if task == "Single Sentence":
    x = batch["input"]
    seq_len = batch["seq_len"]
    y = batch["label"]
    return x, seq_len, y
  elif task == "Similarity and Paraphrase":
    x = jnp.concat([batch["input1"],batch["input2"]])
    seq_len = jnp.concat([batch["seq_len1"],batch["seq_len2"]])
    y = batch["label"]
    return x, seq_len, y



def get_emb(embeddings,x):
  with torch.no_grad():
    x_torch = torch.from_numpy(np.array(x))
    x_emb = embeddings(x_torch)
  x_emb =  jnp.array(x_emb)
  return x_emb










    