# Images

Single frames from one desktop machine (32 GB of RAM, one 8 GB RTX 5050), showing the present state with its
known flaws. They are not comparisons against any other tool. No image contains a third-party photograph.

## The tool and its generated parts (`tools_only/`)

| File | What it shows |
|---|---|
| `08_teeth_gums_front.png` | Generated teeth and gums from the mouth library, as field parts, in the tool's clay view. |
| `09_window_section_mouth_parts.png` | The Field Studio window in its section view, zoomed close: a section plane through a head built from mouth-library parts. The cut shows the parts' layered rings rather than recognizable anatomy; the timeline at the bottom lists the eight build steps (head, upper and lower teeth, upper and lower gums, tongue, mandible, maxilla), and the tool shelf is on the left. |
| `10_window_sphere_shelf.png` | The Field Studio window with a sphere: the tool shelf with names, the layer list ("Implicit field F(X) = 0 (no mesh)"). |
| `11_grids_before_after.png` | The floor grid before and after a rework, on a ball and on a head-and-neck start form. |
| `13_elbow_section_fold_vs_corrective.png` | A section through the synthetic test elbow at 70 degrees, in the plane of the bend (grey = tissue, white = outside). Left: skinning alone; the folded region (red: points with two or more pre-images) is the small area at the upper left of the inner elbow. Right: the contact-solved corrective; no red anywhere. |
| `14_teeth_gums_look.png` | An earlier look render of teeth and gums (layered scattering). It pre-dates a contact fix: note the dark slit between the upper central incisors, and the render noise. |

## The human model, in segments (`model_segments/`)

| File | What it shows |
|---|---|
| `01_eye_right_closeup.png` | The human model's right eye at full resolution: generated iris, sclera with vessels, lashes, lid skin. Known flaws visible: lash roots along the lid margin; the inner corner. |
| `02_eye_left_closeup.png` | The left eye, same render. A different iris from the same seeded generator. |
| `03_lips_closeup.png` | The lips at full resolution. Known flaw, shown on purpose: the lip's colour border steps in about 0.3 mm cells. |
| `04_head_front.png` | Head and hair, front, in the review renderer. The hair is the generated natural base. |
| `05_head_three_quarter.png` | The same, three-quarter view. |
| `06_head_side.png` | The same, side view. |
| `07_hair_back_three_quarter.png` | The hair from behind, after the wave-phase fix (waves on far-apart parts of the head no longer move in step). |
| `12_elbow_55deg_clay.png` | The human model's elbow bent to 55 degrees in clay. The crease shows the present defects. |
| `15_portrait_light_preset_clay.png` | A portrait light preset in the tool's own view, on the clay head. |
| `16_hairline_shadow_before_after_reference.png` | The hairline: hair shadow on skin with the old one-cone shadow, with exact rays, and a 144-ray reference. |
| `17_hand_back_skin.png` | The back of the hand: skin colour with faint veins. |
| `18_forearm_skin.png` | Forearm skin close-up: freckles, a faint vein, generated relief. |


## Release 0.2.0 (`v0.2.0/`): the tool on 2026-11-14

All twelve are Field Studio's own renders and window captures on the same machine, made by the scripted runs described
in the [status review](../docs/STATUS_2026-11-14.md). No test model appears in them.

| File | What it shows |
|---|---|
| `01_window_1920.png` | The window at 1920 x 1080 after the approved interface was built: the four tool categories on the left (Add Objects, Manipulate, Building Tools, Corrective), the tool strip with Transform's settings, the view bar (Clay / Look / Silhouette, X-ray), the Quick keys card, the Scene panel docked on the right and the edge tabs. "Blend" is greyed and marked "coming". |
| `02_milestone_sheet.png` | The milestone session on the card, step by step: open; a sphere from the Shapes row; Swell held and dragged; Settle and Flatten; a Mask stroke and a region Transform; a line Cut through the top; sandstone from the Materials panel; the cut cap swung open 30 degrees and closed again (Transform frames: there is no rig or keyframe system yet); Merge; saved and reopened with the same state hash. |
| `03_milestone_cap_open.png` | One frame of that limited motion in the full window: the sandstone cap open 30 degrees above the body. |
| `04_milestone_reopened.png` | The merged form after save and reopen (its state hash equal to the saved one). This run predates the fix that seals a Merge's contact (status review, section 5.4). |
| `05_brick_whole.png` | Tier 1 of the acceptance ladder: a 215 x 102.5 x 65 mm sandstone brick, its weight (3.30 kg) from the material's density. |
| `06_brick_10cm.png` | The same brick at about 10 cm: grain and the laminae on the side. |
| `07_brick_1cm_grains.png` | At about 1 cm: the grains as relief built at render time from the material, not a texture. |
| `08_brick_arris.png` | The arris (the edge) at about 3 cm. |
| `09_five_bricks_vary.png` | Tier 2: five bricks of one stone made by Array with Vary, each with its own seed (colour, grain, chips). |
| `10_brick_crack.png` | A crack added to one brick (50 % through, 0.7 mm at the mouth): a cavity in the field, visible in section. |
| `11_brick_rougher.png` | One brick with 2.5 times the relief, beside a standard one. |
| `12_materials_panel.png` | The Materials & Generators panel: the library's families and stones, the selected stone's measured properties, the object's size and weight, and the seeded generators (surface, crack, vary across copies). |

Known short-falls, visible here and stated in the review: the stones read cooler and more uniform than reference
photographs of the same stones; chip bites are too smooth up close.
