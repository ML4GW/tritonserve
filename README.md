# `tritonserve`
## Lightweight Python utility for serving a local Triton Inference Server Deployment

Simple utility library that includes a Python context which launches a
Triton Inference Server instance in the background and spins it down upon
context exit. For example:

```python
import tritonserve
import tritonclient.grpc as triton


model_repo_dir = "/path/to/model_repo"
gpus = [0, 1]  # run the server on gpus with ids 0 and 1
with tritonserve(model_repo_dir, gpus=gpus) as instance:
    do_some_prep_work_while_server_spins_up()

    # now wait for the server to come online
    instance.wait()

    # do some inference
    client = triton.InferenceServerClient(url="localhost:8001")
    do_some_inference(client)

# on context exit, the server gets spun down
```
