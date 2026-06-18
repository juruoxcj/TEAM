# TEAM Demo Data

This directory contains six unmodified TCGA BRCA patch images for software testing.

The demo patches are included only to demonstrate the expected file layout and to let users test data loading. They should not be used for scientific evaluation.

Expected layout:

```text
demo_data/
|-- README.md
|-- slide_texts.demo.json
`-- slides/
    `-- TCGA-A7-A13G-01Z-00-DX1.C258C545-8C1F-41D4-846F-962A746CBDFB/
        |-- patch_660.jpg
        |-- patch_661.jpg
        `-- patch_665.jpg
```

Each image should be an RGB patch supported by PIL (`.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`, or `.webp`). Do not place PHI or protected patient identifiers in filenames or metadata.
