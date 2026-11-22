# Data sources and provenance

Raw data are not redistributed. Place the original files below
`ECM_UDE_DATA` (default: `./datasets`) without changing their bytes.

| Dataset | Source | Directory below `ECM_UDE_DATA` |
|---|---|---|
| Panasonic 18650PF | [Mendeley, version 1](https://doi.org/10.17632/wykht8y7tg.1) | `Pan18650/` |
| CALCE A123 and Samsung INR18650-20R | [CALCE battery data](https://calce.umd.edu/data); no dataset-specific DOI identified | `calce_a123_raw/`, `calce_20r_raw/` |
| Replica LFP | [Mendeley, version 1](https://doi.org/10.17632/29kw38kzwj.1) | `mendeley_29kw38kzwj_v1/` |
| LG 18650HG2 | [Mendeley, version 2](https://doi.org/10.17632/b5mj79w5w9.2) | `lg_hg2_v2/extracted/` |
| LG M50T | Pozzato et al., *Data in Brief* 41, 107995, [DOI](https://doi.org/10.1016/j.dib.2022.107995), and [OSF project](https://osf.io/qsabn/) | `stanford_m50t_raw/` |

The [official HG2 version-2 file metadata](https://data.mendeley.com/public-api/datasets/b5mj79w5w9/files?folder_id=root&version=2)
matches the SHA-256 of all three locally extracted root files: `Plot_results.m`
(`036aa647e3995e6dd4d5986576884373a5aaf56c0453ebb3bf6396db65f54533`),
the README (`809c0045eabcf4913a380ad520e4016f52a9c4186e6d15958992df3baf0e1427`),
and the technical PDF (`7af5bfc6156a968316e57d8340005d7403c35d2da0eee3bbd47d8a3d41c347a5`).
These corroborate version 2, but do not establish that the entire downloaded
archive matches the publisher's distribution. The M50T files have matching
cell identifiers; equivalence of the local distribution to the cited OSF
collection has not been independently established.

The HG2 HWFET 25 °C file contains only rest and is replaced by Mixed1 25 °C.
Four HG2 10 °C CSV files are labelled MG1 and are excluded; Mixed3 is absent
at 0 and 10 °C. HG2 Wh values are taken from the CSV files, not MAT.
For the replica LFP cell, the nominal capacity in the label
and measured capacity disagree; some temperature labels and drive-cycle
metadata are inconsistent. See the saved audits for the experimental
selection and caveats.

## Referenced raw files and SHA-256

The following inventory collects traceable raw-file and archive hashes from
the saved experiment provenance and independently checks the local bytes.
An archive's listed members may include unused specimens or lock files; their
presence in the inventory does not imply they were used for fitting.
Paths are relative to `ECM_UDE_DATA`.

| Raw file | SHA-256 |
|---|---|
| `Pan18650/0degC_UDDS_Pan18650PF.mat` | `f43128fcacbd3f2c65955e02a0431be32addd1c0c847677cbe30e9d9054c6aa5` |
| `Pan18650/10degC_UDDS_Pan18650PF.mat` | `971ac1c69feeb81e1f469b6385044660df13cc200f78d204d26ef0f70a7f35bb` |
| `Pan18650/25degC_HWFTa_Pan18650PF.mat` | `333d23b6eb7615d673190ee77d74e54f76ddbdc2b607eae6d1dd57ef6ed0e77b` |
| `Pan18650/25degC_LA92_Pan18650PF.mat` | `808e837526a8c319b9ea5c63466205de900eab6d708445318e9db0a082fe6f31` |
| `Pan18650/25degC_UDDS_Pan18650PF.mat` | `cb9d830727474c17a8eb624bfd16bbf1f4c306c8d32f831ed58ccdb820f0376a` |
| `Pan18650/25degC_US06_Pan18650PF.mat` | `c12a57f8fa2bfcb7fcd30f243781390e1567acb369da647e6bdc08885deb0ac4` |
| `Pan18650/n10degC_UDDS_Pan18650PF.mat` | `51946cb4d4582f15ab43ae3c75b80a3fa2a8c5f653a78327322d35f3e19b8d88` |
| `Pan18650/n20degC_UDDS_Pan18650PF.mat` | `755a3a1aa8614eda9757ef19dedeb8d633ebd951d3c940af20e74b494f89b12f` |
| `calce_20r_raw/extracted/SP1_25C_LC_OCV_11_5_2015/11_5_2015_low current OCV test_SP20-1.xlsx` | `3c43eb4a2fa79256ecb66def693c62e6823a66efd68474d33b5e039ad2247dab` |
| `calce_20r_raw/extracted/SP2_0C_BJDST/02_27_2016_SP20-2_0C_BJDST_80SOC.xls` | `7a65ba974c5ee381a731fde38a578e9467a20072ae53008896530610ec6c6a82` |
| `calce_20r_raw/extracted/SP2_0C_DST/02_24_2016_SP20-2_0C_DST_80SOC.xls` | `8e7963ef0d60e857b4d26b675975e893b07c3b3a3e544b5a01e5650fee5894a1` |
| `calce_20r_raw/extracted/SP2_0C_FUDS/02_25_2016_SP20-2_0C_FUDS_80SOC.xls` | `1deb0e848cbde3520738d20a1cf784886cdf94c4c3bb08451cbe987ecb5e8e3c` |
| `calce_20r_raw/extracted/SP2_0C_US06/02_26_2016_SP20-2_0C_US06_80SOC.xls` | `5a382e0c3f136b0179eb5abf6e1a2a5270e6204e08b4d7e16737a9b538ee2bbe` |
| `calce_20r_raw/extracted/SP2_25C_BJDST/11_12_2015_SP20-2_BJDST_80SOC.xls` | `6ff84c4fabf8897bb03d8574c6fb89069cdc164e8462f22c54b19c00dba677d7` |
| `calce_20r_raw/extracted/SP2_25C_DST/11_05_2015_SP20-2_DST_80SOC.xls` | `0640dc88b48f6192260b706f06b4245320bb607488b0ee7a3f8f9fe5eac026ec` |
| `calce_20r_raw/extracted/SP2_25C_FUDS/11_06_2015_SP20-2_FUDS_80SOC.xls` | `8c24e60e4032af8d4279942141cec74ac34640c3368015f28efa76f989589d33` |
| `calce_20r_raw/extracted/SP2_25C_US06/11_11_2015_SP20-2_US06_80SOC.xls` | `faf52bf0be8b027c0f7abaa0c08a8eccf11b7b566feb5d7739dbd789277edb53` |
| `calce_20r_raw/extracted/SP2_45C_BJDST/12_17_2015_SP20-2_45C_BJDST_80SOC.xls` | `31f383026ee2301856a663857043caf8cf55661a39fa3888664efd544c40daa8` |
| `calce_20r_raw/extracted/SP2_45C_DST/12_11_2015_SP20-2_45C_DST_80SOC.xls` | `bd808f60c1b768bd65bcccd048fce909e38ec83b294cfa3e6bd2542ecd5b9ee5` |
| `calce_20r_raw/extracted/SP2_45C_FUDS/12_15_2015_SP20-2_45C_FUDS_80SOC.xls` | `746fe50a1b399976b125d23a30f0d0bb3c77a79abffd85e6b363145fd455b0e4` |
| `calce_20r_raw/extracted/SP2_45C_US06/12_16_2015_SP20-2_45C_US06_80SOC.xls` | `24d7c25371977a11968bd257c081d8398a4ba2d633474b598d6838a74e36998c` |
| `calce_a123_raw/A123_DST-US06-FUDS-0.zip` | `c18df3909de27c4409d5e9730c5078284f3883ee52193b559ca8e0ee17732505` |
| `calce_a123_raw/A123_DST-US06-FUDS-10.zip` | `c69dbb1569c5757c19d29741472a773a5fc8ab051f0ee76bd1af45ea2a9b27b7` |
| `calce_a123_raw/A123_DST-US06-FUDS-25.zip` | `b7e5402815b491f4052e1fc02ac12723a28a6893e3f695f10a39b773043145f2` |
| `calce_a123_raw/A123_DST-US06-FUDS-40.zip` | `b17af1850a1d508edf6dfb56f92822178bd32c28d65fac3de7f2ff2c0898f2e5` |
| `calce_a123_raw/A123_DST-US06-FUDS-50.zip` | `7fae05b50c7f4edcc799a7d817d56fb701e1e8894b538e043d05d08ab60f91bc` |
| `calce_a123_raw/A123_OCV0-20120618.zip` | `6a933157e8317a01b6157c57c285b6b6f2c5585269aa9ce66ff7e00b0fa9510a` |
| `calce_a123_raw/A123_OCV10-20120611.zip` | `6d53af243558c35b1b796c685a188ab2b82205617d4b7c2b74b838eb31447e9d` |
| `calce_a123_raw/A123_OCV25-20120905.zip` | `3e1877772487b306a14249e80f8db8d28ca03cdba8bfa23b7c1a0f9650dbce3a` |
| `calce_a123_raw/A123_OCV40-20120627.zip` | `095c1bb5bed60c48fdc8b1ab30125e8f3b58b22a1ee95ee2180f6c90f291cd67` |
| `calce_a123_raw/A123_OCV50-20120702.zip` | `28178f47ebc6b9d98fdd7aab11604ba219274a533802a0f36f26ec83c5f8edef` |
| `calce_a123_raw/extracted/DST-US06-FUDS-0/A1-007-DST-US06-FUDS-0-20120813.xlsx` | `bb0e6929ce22dc96bf4fa83e282afb03638d304118c1936520e54a3a5ee68cef` |
| `calce_a123_raw/extracted/DST-US06-FUDS-0/A1-008-DST-US06-FUDS-0-20120813.xlsx` | `0e62afd2c8f00d8d358d4e49fbc94db5ede1dc96ed5aa11143ab583611f22126` |
| `calce_a123_raw/extracted/DST-US06-FUDS-0/~$A1-007-DST-US06-FUDS-0-20120813.xlsx` | `a964ec392e040fdf1e7382cd227ecc6b0f99ed9b12fb4c2e5f5930a2b673675a` |
| `calce_a123_raw/extracted/DST-US06-FUDS-10/A1-007-DST-US06-FUDS-10-20120815.xlsx` | `0fc100fe95b1be04c92e15813ade117a2317688d9d9518f1290adff7b61f9c30` |
| `calce_a123_raw/extracted/DST-US06-FUDS-10/A1-008-DST-US06-FUDS-10-20120815.xlsx` | `895b7c16e6a68cb86c462085c7af949c9ac15e881a213bc604876b67c1f02e1e` |
| `calce_a123_raw/extracted/DST-US06-FUDS-25/A1-007-DST-US06-FUDS-25-20120827.xlsx` | `575f2b0698907cb7514aa24e224476a8b589673e3343aba20fdeca1700fc6b6b` |
| `calce_a123_raw/extracted/DST-US06-FUDS-25/A1-008-DST-US06-FUDS-25-20120827.xlsx` | `c2ef05025c09534ff8ad1aa440456d5282601428dffd3dac775fe47acbae309a` |
| `calce_a123_raw/extracted/DST-US06-FUDS-40/A1-007-DST-US06-FUDS-40-20120822.xlsx` | `f1c66ee149998946bf9bbc928817a16d52d278e6e8312d095658f925d025713c` |
| `calce_a123_raw/extracted/DST-US06-FUDS-40/A1-008-DST-US06-FUDS-40-20120822.xlsx` | `62f7ac3cf804ed71b098f2481551a92caa978bbbaea236aadc98fda5b13698e8` |
| `calce_a123_raw/extracted/DST-US06-FUDS-40/~$A1-008-DST-US06-FUDS-40-20120822.xlsx` | `a964ec392e040fdf1e7382cd227ecc6b0f99ed9b12fb4c2e5f5930a2b673675a` |
| `calce_a123_raw/extracted/DST-US06-FUDS-50/A1-007-DST-US06-FUDS-50-20120824.xlsx` | `549979a13a4d99fbb05fd84ea035b23f529157ae33e5d64c0057e9ff872b2608` |
| `calce_a123_raw/extracted/DST-US06-FUDS-50/A1-008-DST-US06-FUDS-50-20120824.xlsx` | `73efd1713f4ecf45375fbdfc974a7e9af66b973e60d8e4063b7d62f926281560` |
| `calce_a123_raw/extracted/OCV0-20120618/A1-007-OCV0-20120618.xlsx` | `e650baaedc34d3a530d05eb05a0b715d3de7871b8f8e8755daff338642c69a3f` |
| `calce_a123_raw/extracted/OCV0-20120618/A1-008-OCV0-20120618.xlsx` | `861aacc8a8f70ae68d1b761e246cf16e632626b6d7b316de343175fb35c6879d` |
| `calce_a123_raw/extracted/OCV10-20120611/A1-007-OCV10-20120611.xlsx` | `117f2e4fd6e0a0e00a261ef1c9c02338a7b9bc404af46e5a08593b8a35542be2` |
| `calce_a123_raw/extracted/OCV10-20120611/A1-008-OCV10-20120611.xlsx` | `82bdbe30566481aa65365342f99ac6ab806d468f80c61068072b67b11802126d` |
| `calce_a123_raw/extracted/OCV25-20120905/A1-007-OCV-25-20120905.xlsx` | `861c7d51597d93811606d1230f7a5ac60463a2f17470b505c1cff6b9f9bc37ce` |
| `calce_a123_raw/extracted/OCV25-20120905/A1-008-OCV-25-20120905.xlsx` | `aefe51509e728a9e45e68f5bb2d70408bfe2df29d721191795f34cf68fad37bc` |
| `calce_a123_raw/extracted/OCV40-20120627/A1-007-OCV40-20120627.xlsx` | `6539af9683aaf6592f8ede0ea36482a0c330aad8a0e277df82a64195d3cdea98` |
| `calce_a123_raw/extracted/OCV40-20120627/A1-008-OCV40-20120627.xlsx` | `a4481ab34af247028e36eef04bd139c9e07b01ba151c7aea4515ea64781549ee` |
| `calce_a123_raw/extracted/OCV50-20120702/A1-007-OCV50-20120702.xlsx` | `8c9bdb51f7692f77055aeb16bf6fa48c397721fb6e29063164ed6aa395ccff00` |
| `calce_a123_raw/extracted/OCV50-20120702/A1-008-OCV50-20120702.xlsx` | `1a6fcf4d8d22d6b31163b34285557f283f2c10204b3ee7d6795fa064335a367a` |
| `lg_hg2_v2/extracted/0degC/11-29-18_22.00 589_Charge2.mat` | `c38ad40434d2dcd253240e332270ba2ec0ba33e054c7587e140fb740000765ef` |
| `lg_hg2_v2/extracted/0degC/11-30-18_01.23 589_UDDS_0degC_LGHG2.mat` | `a3e4fe8d27d99d5bf12ec60e5016e1145a6b814cfcf2fd591e9ee43c9e7a2568` |
| `lg_hg2_v2/extracted/10degC/11-19-18_23.46 576_Charge2.mat` | `39be5076e4262772fb9882a575bf933fafe1fce9df2b20e3367bc533d91c3a84` |
| `lg_hg2_v2/extracted/10degC/11-20-18_03.10 576_UDDS_10degC_LGHG2.mat` | `f01c711df9063c5a7b844b8dc7d1fe908aeb0fde1d2f5f3b8a7893af697da391` |
| `lg_hg2_v2/extracted/25degC/10-25-18_07.39 549_HPPC_25degC_LGHG2.mat` | `eb3fdfa111a9761d825ce164d8108c09be3129e039eb1fd1c95ba1f3520fcc75` |
| `lg_hg2_v2/extracted/25degC/10-26-18_09.29 549_C20DisCh_25degC_LGHG2.mat` | `570746af29365dbe2aa7850a8b095bf4f131a6692a1c47dc065141dfc789260d` |
| `lg_hg2_v2/extracted/25degC/10-28-18_13.10 551_Cap_1C_25degC_LGHG2.mat` | `0846effa0812b2a1a04e845e220a07ba3653387744bbd8a9a366902adbed2177` |
| `lg_hg2_v2/extracted/25degC/10-28-18_14.16 551_Charge2.mat` | `62b11888092949f3e8fec10888d8cab9334f1eebbb3b6ddec6f2e03cac97bdfa` |
| `lg_hg2_v2/extracted/25degC/10-28-18_17.44 551_UDDS_25degC_LGHG2.mat` | `2beafcb42b5eec07ffa465b95abce1746d90504c440113276dc07f8b8a958b77` |
| `lg_hg2_v2/extracted/25degC/10-29-18_01.36 551_HWFET_25degC_LGHG2.mat` | `28660bc4de7348ba296a855ea778a2078c2764ac78323d2525e2490f571f1f94` |
| `lg_hg2_v2/extracted/25degC/10-29-18_01.47 551_Charge4.mat` | `7241f3c08ca16730e53141daf61e616faa2bda256f55906a59feba533ec19ffc` |
| `lg_hg2_v2/extracted/25degC/10-29-18_03.53 551_LA92_25degC_LGHG2.mat` | `43b9803b858e44ad29e7dcfa59fb395ebb06560732fac86b5fc0c1dc5bb62dcf` |
| `lg_hg2_v2/extracted/25degC/10-29-18_06.42 551_Charge5.mat` | `77baa564542e7c7b242027f64781edbb84dcc5ebe8dac13dd545161964f57f30` |
| `lg_hg2_v2/extracted/25degC/10-29-18_10.08 551_US06_25degC_LGHG2.mat` | `87eb8d7a735e6b54c582f663c21bcc06037390feb773770b9901163878c50485` |
| `lg_hg2_v2/extracted/25degC/10-29-18_11.15 551_Charge6.mat` | `3d69303d040b85c5bd3d19ecd560a3ec354c7f35c16e1b771cec45c9f02059d5` |
| `lg_hg2_v2/extracted/25degC/10-29-18_14.41 551_Mixed1_25degC_LGHG2.mat` | `8e05da70b31c3f8b10d34592cce193daae0478ccabf9112bef59c778cdc3b74d` |
| `lg_hg2_v2/extracted/25degC/551_HWFET.csv` | `5b774a7673d4e624758f59adc6d94efa981390ee728125cccca68846efd77eae` |
| `lg_hg2_v2/extracted/40degC/11-05-18_20.44 556_Charge2.mat` | `b5585e245d5a9e2d8c7ce665d81309035d273cc9cc16ea334c3dc30286f2776f` |
| `lg_hg2_v2/extracted/40degC/11-06-18_00.10 556_UDDS_40degC_LGHG2.mat` | `a38de6d1db2976f96b3b276da91655e5a2dfd71c10f26c78ca370f0f0629bc64` |
| `lg_hg2_v2/extracted/n10degC/12-07-18_20.30 596_Charge2.mat` | `828067a088a0b88c5592b9d154bc9376f8ef54405b42c6c9614da0d2f3e9df9c` |
| `lg_hg2_v2/extracted/n10degC/12-07-18_23.49 596_UDDS_n10degC_LGHG2.mat` | `b9119d610690118aef0558db76005422ce07f65426f1eb7a114bf1cdb7f3afe3` |
| `lg_hg2_v2/extracted/n20degC/12-24-18_11.30 610_UDDS_n20degC_LGHG2.mat` | `30bd81b4ced80a042697f880d634ccf5863731d8fee67cfb17dd78c342abcd76` |
| `lg_hg2_v2/extracted/n20degC/12-25-18_11.00 610_Charge8.mat` | `95e37d201123b0e8f94396f067fb161947fd47900c9b71c12c57d89dcb5784e8` |
| `mendeley_29kw38kzwj_v1/extracted/Battery Test Data/LiFePO4 18650 Rechargeable Cell 3.3V 1100 mAh/25 Degree/1.5C 1C 0.5C.xlsx` | `60c7c4f7d7bb8555de0fcfe133e692d9b087a353584ab335e3ffafae8c09a55f` |
| `mendeley_29kw38kzwj_v1/extracted/Battery Test Data/LiFePO4 18650 Rechargeable Cell 3.3V 1100 mAh/25 Degree/DST FUDS UDDS HPPC WLTS US06.xlsx` | `3883d7248a389a65d5aa3d53e26605df385aa736fdfe9ed0d5b6327119d1a574` |
| `mendeley_29kw38kzwj_v1/extracted/Battery Test Data/LiFePO4 18650 Rechargeable Cell 3.3V 1100 mAh/25 Degree/PCPD.xls` | `9d44621a6009b6b00292c3c12de0481af1c4e65c5e4eba461c1ae1352116e395` |
| `stanford_m50t_raw/G1/INR21700_M50T_T23_Aging_0_05C_G1_Channel_3.xlsx` | `4c43d39fa0c98053407ea6aa7cbac3052c8fd1e2772b5cf38776db17042454c0` |
| `stanford_m50t_raw/G1/INR21700_M50T_T23_Aging_HPPC_G1_Channel_3.xlsx` | `e7787c314e69ace095477aa49cab07d9bc12d7bfb9e360b7c4c995d65727be07` |
| `stanford_m50t_raw/G1/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N100_G1_Part3_Channel_3_Wb_1.xlsx` | `3e4f179fdb12c2774bb260eebd4253afa2364faaecce336b234a5c63da42dbb8` |
| `stanford_m50t_raw/G1/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N100_G1_Part3_Channel_3_Wb_10.xlsx` | `50862a99a44be8d96671e01bf88de31f56c61af2ec85a75a726a259ce539c86e` |
| `stanford_m50t_raw/G1/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N100_G1_Part3_Channel_3_Wb_11.xlsx` | `af032df17113e6f7f02b2f89549d4aa65e6192689edff8b41ad6406a1bd83882` |
| `stanford_m50t_raw/G1/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N100_G1_Part3_Channel_3_Wb_12.xlsx` | `8f425b87da13d8c376bfe6eae0075266e1dd162e3136cc23955ad7ca8b9b795b` |
| `stanford_m50t_raw/G1/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N100_G1_Part3_Channel_3_Wb_13.xlsx` | `85b7d05fefd6ad070ead4efb682317266ec5bf3407033026763d0c0cd2e71d9e` |
| `stanford_m50t_raw/G1/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N100_G1_Part3_Channel_3_Wb_14.xlsx` | `58102db40d8c441904f585cb873372ab3ae4c4d8a4fa8263d4274d0b56669138` |
| `stanford_m50t_raw/G1/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N100_G1_Part3_Channel_3_Wb_15.xlsx` | `e64957a83b4be778b12b9d57706e60e74d4530dfb90624e31ce391e9552ec915` |
| `stanford_m50t_raw/G1/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N100_G1_Part3_Channel_3_Wb_16.xlsx` | `858d7c6534129261e23dc620480994f5ec8d856cc5cee95842807f3e6c8660d8` |
| `stanford_m50t_raw/G1/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N100_G1_Part3_Channel_3_Wb_2.xlsx` | `63a2e70549fa9e112af39e701cb9228756f254b2d368b975e1c77455ff9ce0ba` |
| `stanford_m50t_raw/G1/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N100_G1_Part3_Channel_3_Wb_3.xlsx` | `54edf335a196d4f3b219ee1c42d6399a6344a9be00c7e36a2a853991fac20e5e` |
| `stanford_m50t_raw/G1/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N25_G1_Channel_3.10.xlsx` | `8cf9af7f2d1dda580ff484253620f8d37bbf53f41b4d09c79bc020bc4c712507` |
| `stanford_m50t_raw/G1/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N25_G1_Channel_3.11.xlsx` | `c83c1efcdb790840811abcfdd32f1e2822702ce5a854cf24dc8885e8cfabd5bc` |
| `stanford_m50t_raw/G1/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N25_G1_Channel_3.12.xlsx` | `71b29908d6577b81e0650014e44ecc1baf9e53bd5b6d7b0ef5e21ff007e21851` |
| `stanford_m50t_raw/G1/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N25_G1_Channel_3.13.xlsx` | `675f9ec982dbadc01b6d02f0af7493ca62f1be2c0149df831b39c229630600e6` |
| `stanford_m50t_raw/G1/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N25_G1_Channel_3.14.xlsx` | `a0f57b835b1f0a0b11f3eecdcb2a7a80cc2e66349368161039916fb00c9b82c0` |
| `stanford_m50t_raw/G1/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N25_G1_Channel_3.15.xlsx` | `19500afaa48f2a34357d59a823c1ec23e60885ac272ae39f20f6cd0b4d99ef16` |
| `stanford_m50t_raw/G1/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N25_G1_Channel_3.16.xlsx` | `7a9f52434846127d8851cb1313c80c594f83bbc02967d9b07544d639eedf7d66` |
| `stanford_m50t_raw/G1/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N25_G1_Channel_3.17.xlsx` | `90d72e1fd1ed3ad55150f20c889fadfd44a89194140b080845c99d8644da56e3` |
| `stanford_m50t_raw/G1/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N25_G1_Channel_3.18.xlsx` | `c5f691c9eef0eb127fdbe46769818b52ac71b5d9e78f7ea0382d7230ae8d9538` |
| `stanford_m50t_raw/G1/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N25_G1_Channel_3.19.xlsx` | `7b7bd6259af73f1f6e2f2c07dda64e5e6a6c7aa38b7f5aa075cff5e5687edcb8` |
| `stanford_m50t_raw/V4/INR21700_M50T_T23_Aging_0_05C_V4_Channel_5.xlsx` | `dce90dc165f5bb4f19dc604470e626a670b737874b1fb63187a47df3a1cb6d52` |
| `stanford_m50t_raw/V4/INR21700_M50T_T23_Aging_HPPC_V4_Channel_5.xlsx` | `c415081a8ae981de0a390a7512ee628a1e185d5f15ff192297354a0f8437a2dc` |
| `stanford_m50t_raw/V4/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N25_V4_Channel_1.10.xlsx` | `3b2df25eb149d6e1abc06b813895d1b013cd53b7b14baa1b337d8bbdb3749bd8` |
| `stanford_m50t_raw/V4/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N25_V4_Channel_1.11.xlsx` | `fb313df203612dfd022b045e544a9edc99b2d325c4c7c305944eaa4ef44ebb8c` |
| `stanford_m50t_raw/V4/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N25_V4_Channel_1.12.xlsx` | `f9e82753b702932656d087877432ac3f2f91e0b7307ef7849fa42e0bac53e56d` |
| `stanford_m50t_raw/V4/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N25_V4_Channel_1.13.xlsx` | `16a9ea90178dc7a36e1d17db16ea2180c799934069210803877a29f109aa811f` |
| `stanford_m50t_raw/V4/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N25_V4_Channel_1.14.xlsx` | `acb323fcc1065e009aa6add0777c54a4f307864bb8a2d4ad3cbe6e571f263e15` |
| `stanford_m50t_raw/V4/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N25_V4_Channel_1.15.xlsx` | `bc4909e5ecd1ee0ab71c67350cb9e72d3fd89c178191a2feb6bae05318eb206a` |
| `stanford_m50t_raw/V4/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N25_V4_Channel_1.16.xlsx` | `4933c4ab3ab411c1219f87eb691996d6ad0f87039dc3c3c2fb3ba47cca7357c7` |
| `stanford_m50t_raw/V4/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N25_V4_Channel_1.17.xlsx` | `83cded893c792c8d9816988e92f90a5a69e9d853d8e0aa51594a5a3ddd7a4026` |
| `stanford_m50t_raw/V4/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N25_V4_Channel_1.18.xlsx` | `ad75c10df99ddc546dfe8f449d96a7b58e795ff18c336b8bc24360c76a573951` |
| `stanford_m50t_raw/V4/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N25_V4_Channel_1.19.xlsx` | `6a71c53a1727c98faf803ab74499ab5b92e1e9a57c342834f77b66efe245c63d` |
| `stanford_m50t_raw/V4/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N50_V4_Channel_1.10.xlsx` | `47aa3e67790d0612aaac981fe40be66f80ace27c34f4eaedb4667a8153277eb9` |
| `stanford_m50t_raw/V4/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N50_V4_Channel_1.11.xlsx` | `7b962be92a3e1fdf6ec47b571652731e6e7f3f13f7df930b0a183c8476995ac8` |
| `stanford_m50t_raw/V4/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N50_V4_Channel_1.12.xlsx` | `45f3e8b9fe4ca2e57bbc5da9e661d90f673505f71f7d9adc8e990c7a55a109df` |
| `stanford_m50t_raw/V4/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N50_V4_Channel_1.13.xlsx` | `6622f33eef0094c0cd309b35d70e290f9755db38206e05e9ad095bdc67c43c6b` |
| `stanford_m50t_raw/V4/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N50_V4_Channel_1.14.xlsx` | `f72cffecd0e085e3cd9e9e6e7ebb262bec01fe2f6f4d9a019f4742382e982928` |
| `stanford_m50t_raw/V4/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N50_V4_Channel_1.15.xlsx` | `3920deafb2b12dc5631e1b85692dd913690af37a6930b8b9efc853042fe2a170` |
| `stanford_m50t_raw/V4/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N50_V4_Channel_1.16.xlsx` | `24bb1bf6f39f1fbe5364fecef8f754c14b5b0ff7dd6256c2c029fc5236dcd443` |
| `stanford_m50t_raw/V4/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N50_V4_Channel_1.17.xlsx` | `50f843f23e9a59ea587aa125236561bc978599da565e09231b696b7f456843cd` |
| `stanford_m50t_raw/V4/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N50_V4_Channel_1.18.xlsx` | `2687a5da7440b164e34d656964bf2eb7291827bee116655844cdd1c1d1fceb99` |
| `stanford_m50t_raw/V4/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N50_V4_Channel_1.19.xlsx` | `5e7a3e221c3f430a5c23dc6904d14304e160e6e60f3bae22b9fe290550ff6a4b` |
| `stanford_m50t_raw/V5/INR21700_M50T_T23_Aging_0_05C_V5_Channel_1.xlsx` | `21e04c654fc02a0660b2fc74f4eedeca30095ba4f744764de0fbbbc49462d630` |
| `stanford_m50t_raw/V5/INR21700_M50T_T23_Aging_HPPC_V5_Channel_1.xlsx` | `b36f44ec79b52835571cead5ad13b6d113cd8cddfe2f0b14b3092befa63489aa` |
| `stanford_m50t_raw/V5/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_1C_N25_V5_Channel_1.10.xlsx` | `99d1e01b9de5b29402c42da4cb5db7e67431ae59ce4a6d3c28600285ab8ffa23` |
| `stanford_m50t_raw/V5/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_1C_N25_V5_Channel_1.11.xlsx` | `fd4c0c28eb99fb311b4dcd7efbda9bc0b0fdae0094a2c114d8cbd21527015d38` |
| `stanford_m50t_raw/V5/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_1C_N25_V5_Channel_1.12.xlsx` | `7d27ed4c093afa0323ff1398d31a19044c7e81e939c36a77569519fc7f19ad88` |
| `stanford_m50t_raw/V5/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_1C_N25_V5_Channel_1.13.xlsx` | `901df8b909ad8a2fab64069d6965c1251cb6d9fc9c397da06511f2978f6fe971` |
| `stanford_m50t_raw/V5/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_1C_N25_V5_Channel_1.14.xlsx` | `a8daffef94ce1e491ec550ef7f32d490e2d1ab6deb53c5f9393ee6c989b9062a` |
| `stanford_m50t_raw/V5/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_1C_N25_V5_Channel_1.15.xlsx` | `a3b6ec5c4a84f8a8379bb30832a92019b0372dd9ab8cd9fe9ec63b08cdde9e70` |
| `stanford_m50t_raw/V5/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_1C_N25_V5_Channel_1.16.xlsx` | `6ba4b7365d7a3e09c4b3c2d9682469926cce731367bcb3466aaddb3b53dbec4b` |
| `stanford_m50t_raw/V5/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_1C_N25_V5_Channel_1.17.xlsx` | `6177ef231418fc17e7f1277aca336753c65db4145708266a7efe6d1b301ee332` |
| `stanford_m50t_raw/V5/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_1C_N25_V5_Channel_1.18.xlsx` | `413e09da7d0e3342832ab285dc2e909690f6f4bcdcc03d92a5cd92d54a11350e` |
| `stanford_m50t_raw/V5/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_1C_N25_V5_Channel_1.19.xlsx` | `a35fd3d13709c4a6470e29c76ceacb6cec3e8bb203ad7da3be9fa9b63f521ac4` |
| `stanford_m50t_raw/W10/INR21700_M50T_T23_Aging_0_05C_W10_Channel_6.xlsx` | `9641b06b4899667b5d516b0920cb810a429fa8026541704ede72ec590b9deffc` |
| `stanford_m50t_raw/W10/INR21700_M50T_T23_Aging_HPPC_W10_Channel_6.xlsx` | `4a0f8197c33670ba5541bbca9f20f88b85b9de02c9ca799e6c3a0069055b8864` |
| `stanford_m50t_raw/W10/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N25_W10_Channel_6.10.xlsx` | `4fcd56c84306065b189226c3c4abee00c1332fc7cab9865c424c64b476eea6f7` |
| `stanford_m50t_raw/W10/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N25_W10_Channel_6.11.xlsx` | `c25f50164156c371fe7592bd1e488051db73fa76905a396e94548b198ddbacd7` |
| `stanford_m50t_raw/W10/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N25_W10_Channel_6.12.xlsx` | `2516bc732d87ac15051767d9c0bd48d22e1c674b11a09f1b2cd59bd957488e62` |
| `stanford_m50t_raw/W10/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N25_W10_Channel_6.13.xlsx` | `a1c79be89c5f910c49e06d8633354e4918d5d83e151ffb59be7c89685986da5a` |
| `stanford_m50t_raw/W10/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N25_W10_Channel_6.14.xlsx` | `9ee51a8200afe9dc2579e852627792da7bea658386e459d4f781e811dac49ce0` |
| `stanford_m50t_raw/W10/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N25_W10_Channel_6.15.xlsx` | `915a1dc1ff8f9bbd317b899bbdc800d1625e3ae56c8f6cce740e5f6fb988767c` |
| `stanford_m50t_raw/W10/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N25_W10_Channel_6.16.xlsx` | `8d64d6607470f2d4c91de3cb469b55d0dead14bed11a005db2e9986e4b75da20` |
| `stanford_m50t_raw/W10/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N25_W10_Channel_6.17.xlsx` | `3a3ee589dc26ca9b7f30cbccdee95f73941902d8293628197058ea8ee09ad635` |
| `stanford_m50t_raw/W10/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N25_W10_Channel_6.18.xlsx` | `98ea277f5be0f1b8137980d9d677995b3565d50f8e911793787b8cd9afa9b76b` |
| `stanford_m50t_raw/W10/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N25_W10_Channel_6.19.xlsx` | `5472c95c56d9f59b725edc44dc554229e5fecd4a37b45db097d9d58649a945b6` |
| `stanford_m50t_raw/W10/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N75_W10_Channel_6.10.xlsx` | `792a3121f76a5dcde802dc395b00e3c530a06fa13bcc62ca59df2713464c04de` |
| `stanford_m50t_raw/W10/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N75_W10_Channel_6.11.xlsx` | `c40bb4a3660b7279c3f6c5e1c901235ab91164a1009c46b903e51ce87e118ec5` |
| `stanford_m50t_raw/W10/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N75_W10_Channel_6.12.xlsx` | `6fab54fd4a96b1139a82f384e0e1b645a4c1482dc4ca619bf5ab9cf2adfa2369` |
| `stanford_m50t_raw/W10/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N75_W10_Channel_6.13.xlsx` | `ccdd8965ea713d3fcfed9ee16efef57bf8624c153727e91d45d5ddee25c96c47` |
| `stanford_m50t_raw/W10/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N75_W10_Channel_6.14.xlsx` | `73bbd25de54d4f3494a419a6e631aa78a0e4ea28e4e133b1286e1895f72c69d3` |
| `stanford_m50t_raw/W10/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N75_W10_Channel_6.15.xlsx` | `bf84e57c732e0c38fdac1ce4827e7029ba68577f00b67a9dc18c6a7d65074839` |
| `stanford_m50t_raw/W10/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N75_W10_Channel_6.16.xlsx` | `5a740f6b3f81045513880bdea57dc8b30d6df92b51522e0d0d803da1f11fb634` |
| `stanford_m50t_raw/W10/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N75_W10_Channel_6.17.xlsx` | `93381f4295ba9746cd35fdff8e7b4ca098086d8ee9c4cf2b35d20ff7f069ee34` |
| `stanford_m50t_raw/W10/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N75_W10_Channel_6.18.xlsx` | `df0a71e3137a97b90ac0c4d409f4eee35e46af695a67e4c045970ca2d14116f7` |
| `stanford_m50t_raw/W10/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N75_W10_Channel_6.19.xlsx` | `dd1229a30a2a7da6d3b044daa2cc3d0ea229cabac3b94cfbc9805340a5354dcd` |
| `stanford_m50t_raw/W3/INR21700_M50T_T23_Aging_0_05C_W3_Channel_3.xlsx` | `002ac495004d3d870d2b6cd6744e47cc1976e2dc828fb66311ef8ac6e3f5aff1` |
| `stanford_m50t_raw/W3/INR21700_M50T_T23_Aging_HPPC_W3_Channel_4.xlsx` | `0de4e1dda2914d005025f9339077a869c1175d9da547667eb08a663e2da457d8` |
| `stanford_m50t_raw/W3/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N25_W3_Channel_3.10.xlsx` | `0582fb5b5a5d7fec5ab761690ac1b62005a531eaf6f7a8563e10b367c074ef33` |
| `stanford_m50t_raw/W3/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N25_W3_Channel_3.11.xlsx` | `3bc179de256509ddf8ab51f07fcbaa92debb7a5a2368110df02380617249dc2e` |
| `stanford_m50t_raw/W3/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N25_W3_Channel_3.12.xlsx` | `9fdd01f88eb1725116797cf8a13cf6e35e16f2d16aad06ddb7c7c03b931643a1` |
| `stanford_m50t_raw/W3/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N25_W3_Channel_3.13.xlsx` | `27f81fb910bf6784dd6819f5ab6bb803bec8584b6285fb0ef6a420206e3506b8` |
| `stanford_m50t_raw/W3/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N25_W3_Channel_3.14.xlsx` | `d7e9f46c70d516d3730b5e049d4c0327488b68b9dc2ab2c4c72fd608fb04d1a1` |
| `stanford_m50t_raw/W3/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N25_W3_Channel_3.15.xlsx` | `bcd4956ced396b66440ed9e405b6c7036c756624ceae3ace608bee7f5e1de389` |
| `stanford_m50t_raw/W3/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N25_W3_Channel_3.16.xlsx` | `1a9675428cc76923259c21aa952fb65e79b38d0dc5cdbcd2c959f7532d4e4b78` |
| `stanford_m50t_raw/W3/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N25_W3_Channel_3.17.xlsx` | `624257b5fc6c21c484a794ce89955772a51adfd28b64ddeba97ba2843a912632` |
| `stanford_m50t_raw/W3/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N25_W3_Channel_3.18.xlsx` | `1cd75f46959bd1c1b4bbac90437d9c57c38bcd7aa9b38a739a63770cd6a0dd4b` |
| `stanford_m50t_raw/W3/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N25_W3_Channel_3.19.xlsx` | `bf7bebcb6aefa20382918501e1d0504df91fd20ff64249a9b134bb6acfab7249` |
| `stanford_m50t_raw/W3/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N75_W3_Channel_3.10.xlsx` | `5177ac2f62c7885b1e5243b0fb0a68887a433aa9ce9f15079469e40701dcf73b` |
| `stanford_m50t_raw/W3/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N75_W3_Channel_3.11.xlsx` | `8aeec93b90e48f7634b6b65c25c88ac2db20b5d9cfa09ebead0d42e0d859e66a` |
| `stanford_m50t_raw/W3/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N75_W3_Channel_3.12.xlsx` | `a34e7274aabef495fa2152005e1f02410d8959d64241f810a69dafd2318d2cd9` |
| `stanford_m50t_raw/W3/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N75_W3_Channel_3.13.xlsx` | `2280287bd6c52af4499fc27e6ad8beb64052a200fc51807d7319ceaf0e9bc958` |
| `stanford_m50t_raw/W3/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N75_W3_Channel_3.14.xlsx` | `7579e0c31021dbd74e024541c2574267691907926177532621cf9ab585ca1c03` |
| `stanford_m50t_raw/W3/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N75_W3_Channel_3.15.xlsx` | `915df438b25870307e4cb2fdc71c06473b29ff345981acecb45ef16303efedcf` |
| `stanford_m50t_raw/W3/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N75_W3_Channel_3.16.xlsx` | `210aa85299c44efd6d65dd643d42c72a8e7306cef0dac211b4e1c0f3e0d581bd` |
| `stanford_m50t_raw/W3/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N75_W3_Channel_3.17.xlsx` | `a443a4a651800c11145972031a9515ae20cc5249e84ffb43286e0d6d57c7b6ec` |
| `stanford_m50t_raw/W3/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N75_W3_Channel_3.18.xlsx` | `9ce4cee35951389d0357f1c18435bab5b5e5abc53b340c1a3198f57abb44d4b5` |
| `stanford_m50t_raw/W3/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N75_W3_Channel_3.19.xlsx` | `8ff5a86df3154d1af3a0c44f7507400c79ab932ea42ffd07cd8c9290da114690` |
| `stanford_m50t_raw/W7/INR21700_M50T_T23_Aging_0_05C_W7_Channel_3.xlsx` | `fd83ee4b689c6e41d56db70787d9e939f703a53e33a188ea6c28c1f0ee15f27b` |
| `stanford_m50t_raw/W7/INR21700_M50T_T23_Aging_HPPC_W7_Channel_3.xlsx` | `41cd3422cd092e94b12abcd5cb8ead45763bdc601215666fdb215c90351616f0` |
| `stanford_m50t_raw/W7/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N125_W7_Channel_3_Wb_10.xlsx` | `046457f926843e8f75fc6ce6f1e00d656b195b2f17b63f6bb17de116794d8afd` |
| `stanford_m50t_raw/W7/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N25_W7_Channel_3.10.xlsx` | `e591ec3bca623c46870b80dacea47f47b1175b8d9e2b0ee040e5c4de30a68006` |
| `stanford_m50t_raw/W7/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N25_W7_Channel_3.11.xlsx` | `17f3a7c81daa24569fcafd5becf343449eda234c577e0849b47553769d88f3f6` |
| `stanford_m50t_raw/W7/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N25_W7_Channel_3.12.xlsx` | `b3bc25848155ba70c829e2e0250586b21c8f5048d5a6c993de20352cfde097cb` |
| `stanford_m50t_raw/W7/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N25_W7_Channel_3.13.xlsx` | `15a6ab878be078120f799c83f5676defca7183d1ecb2df790845a1afb8bd34f8` |
| `stanford_m50t_raw/W7/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N25_W7_Channel_3.14.xlsx` | `6295a99affedc43ea8641434ebf0586edc9414b8ba762d187562f623220ed552` |
| `stanford_m50t_raw/W7/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N25_W7_Channel_3.15.xlsx` | `27f802ad41f668d6f7e2b85a8f61de1047ad434a8756d4a3f8b02f0863298050` |
| `stanford_m50t_raw/W7/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N25_W7_Channel_3.16.xlsx` | `6974785925cc12bb504a53c08c7464dd52176f6cbe2f3e2cb5a875e1678dfa80` |
| `stanford_m50t_raw/W7/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N25_W7_Channel_3.17.xlsx` | `aeb0739fe21a3ee0d145d2c5fbd6dae82264cffe19b55fa60e239fe918458518` |
| `stanford_m50t_raw/W7/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N25_W7_Channel_3.18.xlsx` | `fe3b0ab6541e074512f96fd8a49d3c2c6841d97f0e51a267ee917e7ef4398d47` |
| `stanford_m50t_raw/W7/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N25_W7_Channel_3.19.xlsx` | `a9537ab0f729c251eafbcdc1ea5d40339f05f1f65cae3b4e4439db397233dda4` |
| `stanford_m50t_raw/W7/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N75_W7_Channel_3.10.xlsx` | `131b915ffe1f582c810a3f13c7525702bf22e44794d08ae8abf26ce1c6537b2d` |
| `stanford_m50t_raw/W7/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N75_W7_Channel_3.11.xlsx` | `fea5b93518d1aa1ca4d01a411d14811b939a6f5510dcc5270ef56eec96aaa8d3` |
| `stanford_m50t_raw/W7/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N75_W7_Channel_3.12.xlsx` | `3d7bd5b23874b1a9fbea3afc31cd5277692be2323831bc2de4879e432e5583f6` |
| `stanford_m50t_raw/W7/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N75_W7_Channel_3.13.xlsx` | `7cee492eb8e64a99b12506b31b8a9258650611b67b4e6753ff9f9719837306ae` |
| `stanford_m50t_raw/W7/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N75_W7_Channel_3.14.xlsx` | `98bbec6b52ab17b88677bd667b787fa37563335a4a7627e4233383b65456a7e9` |
| `stanford_m50t_raw/W7/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N75_W7_Channel_3.15.xlsx` | `b6951c0fd92f54b91a4a1e74edfdb98669ad102a18714c20dd24852332cc9011` |
| `stanford_m50t_raw/W7/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N75_W7_Channel_3.16.xlsx` | `5135eb631466bc2c14887e3626897b9ca9ca2fde6b635dff7f3a7a0588c63f33` |
| `stanford_m50t_raw/W7/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N75_W7_Channel_3.17.xlsx` | `23e90a043aab2557c0eb463e5f5d95e07b34a9c5d635b9d6a150087e66d82c9c` |
| `stanford_m50t_raw/W7/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N75_W7_Channel_3.18.xlsx` | `2a1df33e7c279611cb53447b4c921a4e4c44306f2851afa498823c10f92b8ab8` |
| `stanford_m50t_raw/W7/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_25C_N75_W7_Channel_3.19.xlsx` | `77c3be9e61664dffb70461bb73547db2930673a82400e3b022be3ff74ba42fbb` |
| `stanford_m50t_raw/W8/INR21700_M50T_T23_Aging_0_05C_W8_Channel_4.xlsx` | `9a809e018231a366cb98d4843f4ad672c6949592dc796491d0bcd4b578cbd5c8` |
| `stanford_m50t_raw/W8/INR21700_M50T_T23_Aging_HPPC_W8_Channel_4.xlsx` | `8f40ce1b32a91d536e4e9f86a6e8ed18bac26fc31812ecfc2c87d0f444fee55c` |
| `stanford_m50t_raw/W8/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_5C_N25_W8_Channel_4.10.xlsx` | `1916f6683cf55b67cf52e4b63e16e5f483166039bfe878e399963b5af16f114b` |
| `stanford_m50t_raw/W8/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_5C_N25_W8_Channel_4.11.xlsx` | `2225652314f845907f51e1ad7a2496fb3784983f736938bdb80ffa0363385c29` |
| `stanford_m50t_raw/W8/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_5C_N25_W8_Channel_4.12.xlsx` | `fb36da6aee45cbf39ae4b879082d0de1a8e1eae43b0635302ac770b80f1822a8` |
| `stanford_m50t_raw/W8/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_5C_N25_W8_Channel_4.13.xlsx` | `9eb6b992587f648829b2265d8e1a7179309b308fd2b572fef3ccfdfb00e7931c` |
| `stanford_m50t_raw/W8/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_5C_N25_W8_Channel_4.14.xlsx` | `fd16e10fb7bebd15d9cd627d3d5d9a10d06558fb20f981665fb6d505ad49cfa0` |
| `stanford_m50t_raw/W8/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_5C_N25_W8_Channel_4.15.xlsx` | `72f6a2ad08fbfd5e3c4815d45e79f14657bfed62626b6c8ebf52bc95512dcb1f` |
| `stanford_m50t_raw/W8/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_5C_N25_W8_Channel_4.16.xlsx` | `cf57fea9210fb03f99e6b0a6cd63dc8b04fd2affc4af78b661cf9fd884698579` |
| `stanford_m50t_raw/W8/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_5C_N25_W8_Channel_4.17.xlsx` | `d9da6be64e8f931ca536ba8bad2291eedf2d8c9ce5d39d3c0b0b2c60f4b04ea6` |
| `stanford_m50t_raw/W8/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_5C_N25_W8_Channel_4.18.xlsx` | `bbdf7e1ed216c9683cf34b5ed30a1d2bdd21dfa277a9d0a1c7f72c7a0c3628ce` |
| `stanford_m50t_raw/W8/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_5C_N25_W8_Channel_4.19.xlsx` | `0871c4ef81e32cfe1a2fceae7852ecec7d8c263ef5fbdccf81b6bdc7047e8c93` |
| `stanford_m50t_raw/W8/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_5C_N75_W8_Channel_4.10.xlsx` | `84f613b35f767cf615f9e169665d280995fca2dfdb04c5e62cf13d488d6f7033` |
| `stanford_m50t_raw/W8/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_5C_N75_W8_Channel_4.11.xlsx` | `42400852704578cbc69ffb75f739cd9613179bec355f93de51873fa63801fee6` |
| `stanford_m50t_raw/W8/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_5C_N75_W8_Channel_4.12.xlsx` | `191ae997fc1e9a7f5cd56e0732b3f5bb40220c56e0de1d6ac1c520132e26e0ae` |
| `stanford_m50t_raw/W8/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_5C_N75_W8_Channel_4.13.xlsx` | `b4adf6f675007c14eae023beb5d0d390c19ebd0c4d25ae1995ef0755adb637a2` |
| `stanford_m50t_raw/W8/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_5C_N75_W8_Channel_4.14.xlsx` | `c0fc0850b1ee0545fe5b6b48071d88a78b0ae1c9bfa1779865e6e92b5a478cc3` |
| `stanford_m50t_raw/W8/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_5C_N75_W8_Channel_4.15.xlsx` | `845a4bda24b58c2290fd9060d04bfe7796059c10cfae984e7d8087b603e69d98` |
| `stanford_m50t_raw/W8/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_5C_N75_W8_Channel_4.16.xlsx` | `e5dcfc596f955f57a4f171753d0b1d9e920296564993aa71410ce5b3e8968a2d` |
| `stanford_m50t_raw/W8/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_5C_N75_W8_Channel_4.17.xlsx` | `5c28a8fb061242f958d3d449eb20c6445b9f5f8f1d34f61f874295f3a990af8c` |
| `stanford_m50t_raw/W8/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_5C_N75_W8_Channel_4.18.xlsx` | `a4e307112e45a37e23f274f078b6f6dfaafa8ece9b8b07b567e1b563b16ea812` |
| `stanford_m50t_raw/W8/INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_5C_N75_W8_Channel_4.19.xlsx` | `f7b892d5792c21d1e7c87dcc07ab2589b35ed0219ad2a4ce1e6e9975b94d22bc` |
