from Spoofing3D.adv_utils.poster2img import put_poster_on_batch


def put_poster_on_batch_inputs_eval(leaning_poster, batch_inputs,mask_aug=False):
    put_poster_on_batch(batch_inputs,leaning_poster,ev=True,mask_aug=False)