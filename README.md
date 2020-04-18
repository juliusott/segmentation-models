# segmentation-models
Comparing some Segmentation Models on the BDD100k Dataset for Autonomous Driving

## Link to the dataset
https://bdd-data.berkeley.edu/

### Output examples 
|model          | Input     | Input Size    | Output        | Out Size  |
|-----          |-------    |-------------- |--------       |------     |
|Deeplabv3      |![input]   | (1280,720)    |![Deeplab]     |(640,360)  |
|SegNet         |![input2]  | (1280,720)    |![SegNet]      |(640,360)  |
|SegResNet      |![input2]  | (1280,720)    |![SegResNet]   |(640,360)  |
|Autoencoder    |![input]   | (1280,720)    |![Auto]        |(640,360)  |


[input]: /results/input.png
[input2]: /results/input2.png
[Deeplab]: results/segmented.png 
[SegNet]: /results/segmented_by_segnet.png
[SegResNet]: /results/segmented_by_segresnet.png
[Auto]: /results/reconstructed.png

