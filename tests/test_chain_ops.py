"""
Comprehensive tests for curaster Chain operation combinations.

Each test exercises a different chain pattern and verifies:
  - correct output band count
  - correct output dimensions
  - correct pixel values (where analytically predictable)
  - that ordering of clip/reproject is respected strictly

Tests are grouped by the primary operation under test.
"""

import os
import json
import pytest
import numpy as np
from osgeo import gdal

import curaster
from conftest import read_tif_as_array, W, H, BANDS, EPSG_SRC, EPSG_DST, GT

TARGET_CRS = f"EPSG:{EPSG_DST}"
NODATA     = -9999.0


# ── helpers ──────────────────────────────────────────────────────────────────

def _result_to_array(res) -> np.ndarray:
    """Convert a RasterResult to a (bands, H, W) numpy array."""
    arr = np.array(res.data())
    return arr.reshape(res.bands, res.height, res.width)


def _collect_iter(chain) -> np.ndarray:
    """Drain an iter_begin() queue and reconstruct the full (bands, H, W) array.

    Chunks are sorted by y_offset before assembly because OMP parallel execution
    delivers them in completion order, not spatial order.
    """
    chunks = []
    q = chain.iter_begin(buf_chunks=2)
    while True:
        c = q.next()
        if c is None:
            break
        chunks.append(c)
    if not chunks:
        return None
    chunks.sort(key=lambda c: c["y_offset"])
    # Derive band count from data shape: (H,W) = 1 band, (nb,H,W) = nb bands
    d0 = np.asarray(chunks[0]["data"])
    nb  = 1 if d0.ndim == 2 else d0.shape[0]
    w   = chunks[0]["width"]
    h   = sum(c["height"] for c in chunks)
    out = np.empty((nb, h, w), dtype=np.float32)
    y   = 0
    for c in chunks:
        ch  = c["height"]
        arr = np.asarray(c["data"], dtype=np.float32).reshape(nb, ch, w)
        out[:, y:y + ch, :] = arr
        y  += ch
    return out


def _band_count_from_file(path: str) -> int:
    ds = gdal.Open(path)
    n  = ds.RasterCount
    ds = None
    return n


def _dims_from_file(path: str):
    ds = gdal.Open(path)
    w, h = ds.RasterXSize, ds.RasterYSize
    ds   = None
    return w, h


# ── passthrough / select_bands ────────────────────────────────────────────────

class TestPassthrough:
    def test_to_memory_single_band(self, tif_1band):
        res = curaster.open(tif_1band).to_memory()
        assert res.bands == 1
        assert res.width == W and res.height == H

    def test_to_memory_multiband(self, tif_4band):
        res = curaster.open(tif_4band).to_memory()
        assert res.bands == BANDS
        assert res.width == W and res.height == H
        arr = _result_to_array(res)
        for b in range(BANDS):
            np.testing.assert_allclose(arr[b], (b + 1) * 10.0, rtol=1e-5)

    def test_save_local_multiband(self, tif_4band, tmp_path):
        out = str(tmp_path / "out_pass.tif")
        curaster.open(tif_4band).save_local(out)
        assert _band_count_from_file(out) == BANDS

    def test_iter_begin_multiband(self, tif_4band):
        arr = _collect_iter(curaster.open(tif_4band))
        assert arr.shape[0] == BANDS
        assert arr.shape[1] == H and arr.shape[2] == W

    def test_select_bands_to_memory(self, tif_4band):
        res = curaster.open(tif_4band, bands=[2, 4]).to_memory()
        assert res.bands == 2
        arr = _result_to_array(res)
        np.testing.assert_allclose(arr[0], 20.0, rtol=1e-5)   # band 2
        np.testing.assert_allclose(arr[1], 40.0, rtol=1e-5)   # band 4

    def test_select_bands_save_local(self, tif_4band, tmp_path):
        out = str(tmp_path / "out_sel.tif")
        curaster.open(tif_4band, bands=[1, 3]).save_local(out)
        assert _band_count_from_file(out) == 2


# ── clip ──────────────────────────────────────────────────────────────────────

class TestClip:
    def test_clip_single_band(self, tif_1band, aoi_src):
        res = curaster.open(tif_1band).clip(aoi_src).to_memory()
        assert res.bands == 1
        arr = _result_to_array(res)
        # pixels inside AOI keep their value; outside become 0/nodata
        assert arr.max() > 0

    def test_clip_multiband_to_memory(self, tif_4band, aoi_src):
        res = curaster.open(tif_4band).clip(aoi_src).to_memory()
        assert res.bands == BANDS

    def test_clip_multiband_save_local(self, tif_4band, aoi_src, tmp_path):
        out = str(tmp_path / "out_clip.tif")
        curaster.open(tif_4band).clip(aoi_src).save_local(out)
        assert _band_count_from_file(out) == BANDS

    def test_clip_multiband_iter_begin(self, tif_4band, aoi_src):
        arr = _collect_iter(curaster.open(tif_4band).clip(aoi_src))
        assert arr.shape[0] == BANDS

    def test_clip_masks_border_pixels(self, tif_4band, aoi_src):
        """Top-left corner pixel should be masked (outside the 20%-inset AOI)."""
        res = curaster.open(tif_4band).clip(aoi_src).to_memory()
        arr = _result_to_array(res)
        assert arr[0, 0, 0] == pytest.approx(0.0, abs=1e-5)

    def test_clip_preserves_interior_pixels(self, tif_4band, aoi_src):
        """Centre pixel should have its original value after clip."""
        res = curaster.open(tif_4band).clip(aoi_src).to_memory()
        arr = _result_to_array(res)
        cy, cx = H // 2, W // 2
        np.testing.assert_allclose(arr[0, cy, cx], 10.0, rtol=1e-4)


# ── algebra ──────────────────────────────────────────────────────────────────

class TestAlgebra:
    def test_algebra_produces_1_band(self, tif_4band):
        res = curaster.open(tif_4band).algebra("B1 + B2").to_memory()
        assert res.bands == 1

    def test_algebra_correct_values(self, tif_4band):
        # band1=10, band2=20 → B1+B2 = 30
        res = curaster.open(tif_4band).algebra("B1 + B2").to_memory()
        arr = _result_to_array(res)
        np.testing.assert_allclose(arr[0], 30.0, rtol=1e-5)

    def test_ndvi_formula(self, tif_4band):
        # band4=40, band3=30 → (B4-B3)/(B4+B3) = 10/70 ≈ 0.1429
        res = curaster.open(tif_4band).algebra("(B4 - B3) / (B4 + B3)").to_memory()
        arr = _result_to_array(res)
        np.testing.assert_allclose(arr[0], 10.0 / 70.0, rtol=1e-4)

    def test_algebra_after_clip(self, tif_4band, aoi_src):
        res = curaster.open(tif_4band).clip(aoi_src).algebra("B1 + B2").to_memory()
        assert res.bands == 1

    def test_algebra_save_local(self, tif_4band, tmp_path):
        out = str(tmp_path / "out_alg.tif")
        curaster.open(tif_4band).algebra("B4 - B3").save_local(out)
        assert _band_count_from_file(out) == 1


# ── reproject ─────────────────────────────────────────────────────────────────

class TestReproject:
    def test_reproject_multiband_to_memory(self, tif_4band):
        res = curaster.open(tif_4band).reproject(TARGET_CRS).to_memory()
        assert res.bands == BANDS

    def test_reproject_changes_crs(self, tif_4band):
        info_src = curaster.open(tif_4band).get_info()
        info_dst = curaster.open(tif_4band).reproject(TARGET_CRS).get_info()
        assert str(EPSG_DST) in info_dst["crs"], (
            f"Expected EPSG:{EPSG_DST} in reprojected CRS, got: {info_dst['crs']}"
        )
        assert info_dst["crs"] != info_src["crs"]

    def test_reproject_save_local_multiband(self, tif_4band, tmp_path):
        out = str(tmp_path / "out_reproj.tif")
        curaster.open(tif_4band).reproject(TARGET_CRS).save_local(out)
        assert _band_count_from_file(out) == BANDS

    def test_reproject_iter_begin_multiband(self, tif_4band):
        arr = _collect_iter(curaster.open(tif_4band).reproject(TARGET_CRS))
        assert arr.shape[0] == BANDS


# ── clip + reproject ordering ─────────────────────────────────────────────────

class TestClipReprojectOrdering:
    """
    .clip().reproject() vs .reproject().clip() MUST behave differently.

    clip().reproject() – the GeoJSON polygon is in source CRS (EPSG:4326):
        clip masks pixels in source space, then the masked raster is warped.

    reproject().clip() – the GeoJSON polygon is in target CRS (EPSG:32632):
        the full raster is warped first, then the clip is applied in target space.

    When we pass the *same* source-CRS GeoJSON polygon to both chains, the
    second chain interprets lat/lon values as UTM metres → almost everything
    is masked because lat/lon values like 10.0–11.0 are tiny UTM offsets.
    The two outputs must therefore differ.
    """

    def test_clip_then_reproject_runs(self, tif_4band, aoi_src):
        """clip().reproject() completes without error and preserves band count."""
        res = curaster.open(tif_4band).clip(aoi_src).reproject(TARGET_CRS).to_memory()
        assert res.bands == BANDS

    def test_reproject_then_clip_runs(self, tif_4band, aoi_src):
        """reproject().clip() completes without error and preserves band count."""
        res = curaster.open(tif_4band).reproject(TARGET_CRS).clip(aoi_src).to_memory()
        assert res.bands == BANDS

    def test_ordering_produces_different_results(self, tif_4band, aoi_src):
        """
        Using the same source-CRS GeoJSON for both orderings must yield
        different pixel values because the clip geometry is interpreted in
        different coordinate systems.
        """
        res_cr = curaster.open(tif_4band).clip(aoi_src).reproject(TARGET_CRS).to_memory()
        res_rc = curaster.open(tif_4band).reproject(TARGET_CRS).clip(aoi_src).to_memory()
        arr_cr = _result_to_array(res_cr)
        arr_rc = _result_to_array(res_rc)
        # The arrays may have different sizes; compare nonzero pixel counts.
        nz_cr = np.count_nonzero(arr_cr[0])
        nz_rc = np.count_nonzero(arr_rc[0])
        assert nz_cr != nz_rc, (
            f"clip().reproject() and reproject().clip() produced identical results "
            f"(nonzero pixels: {nz_cr}); ordering is not being respected."
        )

    def test_clip_then_reproject_correct_crs_aoi(self, tif_4band, aoi_src):
        """
        clip().reproject() with source-CRS AOI should preserve a substantial
        fraction of interior pixels (60% inset AOI covers 36% of image area).
        """
        res = curaster.open(tif_4band).clip(aoi_src).reproject(TARGET_CRS).to_memory()
        arr = _result_to_array(res)
        total = arr.shape[1] * arr.shape[2]
        nonzero = np.count_nonzero(arr[0])
        assert nonzero / total > 0.20, (
            f"Expected >20% non-masked pixels after clip().reproject() with "
            f"source-CRS AOI, got {nonzero/total:.1%}"
        )

    def test_reproject_then_clip_correct_crs_aoi(self, tif_4band, aoi_dst):
        """
        reproject().clip() with target-CRS AOI should also preserve a
        substantial fraction of the warped image.
        """
        res = curaster.open(tif_4band).reproject(TARGET_CRS).clip(aoi_dst).to_memory()
        arr = _result_to_array(res)
        total = arr.shape[1] * arr.shape[2]
        nonzero = np.count_nonzero(arr[0])
        assert nonzero / total > 0.10, (
            f"Expected >10% non-masked pixels after reproject().clip() with "
            f"target-CRS AOI, got {nonzero/total:.1%}"
        )

    def test_clip_then_reproject_save_local(self, tif_4band, aoi_src, tmp_path):
        out = str(tmp_path / "out_cr.tif")
        curaster.open(tif_4band).clip(aoi_src).reproject(TARGET_CRS).save_local(out)
        assert _band_count_from_file(out) == BANDS

    def test_reproject_then_clip_save_local(self, tif_4band, aoi_src, tmp_path):
        out = str(tmp_path / "out_rc.tif")
        curaster.open(tif_4band).reproject(TARGET_CRS).clip(aoi_src).save_local(out)
        assert _band_count_from_file(out) == BANDS


# ── focal (neighborhood) ─────────────────────────────────────────────────────

class TestFocal:
    def test_focal_single_band_to_memory(self, tif_1band):
        res = curaster.open(tif_1band).focal().to_memory()
        assert res.bands == 1

    def test_focal_multiband_to_memory(self, tif_4band):
        """Each input band should produce one independent focal output band."""
        res = curaster.open(tif_4band).focal().to_memory()
        assert res.bands == BANDS

    def test_focal_multiband_save_local(self, tif_4band, tmp_path):
        out = str(tmp_path / "out_focal.tif")
        curaster.open(tif_4band).focal().save_local(out)
        assert _band_count_from_file(out) == BANDS

    def test_focal_multiband_iter_begin(self, tif_4band):
        arr = _collect_iter(curaster.open(tif_4band).focal())
        assert arr.shape[0] == BANDS

    def test_focal_mean_of_uniform_preserves_value(self, tif_4band):
        """focal(mean) on a uniform raster must return the same constant."""
        res = curaster.open(tif_4band).focal(stat="mean").to_memory()
        arr = _result_to_array(res)
        # Interior pixels (away from border) must equal input value
        for b in range(BANDS):
            interior = arr[b, 2:-2, 2:-2]
            np.testing.assert_allclose(interior, (b + 1) * 10.0, rtol=1e-4)

    # ── focal + clip (THE BUG-FIX CASES) ─────────────────────────────────────

    def test_focal_clip_to_memory_band_count(self, tif_4band, aoi_src):
        """focal().clip() must return N bands, not 1."""
        res = curaster.open(tif_4band).focal().clip(aoi_src).to_memory()
        assert res.bands == BANDS, (
            f"focal().clip().to_memory() returned {res.bands} bands; expected {BANDS}"
        )

    def test_focal_clip_save_local_band_count(self, tif_4band, aoi_src, tmp_path):
        """focal().clip().save_local() must write N bands to disk, not 1."""
        out = str(tmp_path / "out_focal_clip.tif")
        curaster.open(tif_4band).focal().clip(aoi_src).save_local(out)
        n = _band_count_from_file(out)
        assert n == BANDS, (
            f"focal().clip().save_local() wrote {n} bands; expected {BANDS}"
        )

    def test_focal_clip_iter_begin_band_count(self, tif_4band, aoi_src):
        """focal().clip() streaming must yield N-band chunks."""
        arr = _collect_iter(curaster.open(tif_4band).focal().clip(aoi_src))
        assert arr.shape[0] == BANDS

    def test_focal_clip_masks_border(self, tif_4band, aoi_src):
        """After focal().clip(), top-left corner should be masked."""
        res = curaster.open(tif_4band).focal().clip(aoi_src).to_memory()
        arr = _result_to_array(res)
        assert arr[0, 0, 0] == pytest.approx(0.0, abs=1e-5)

    # ── focal + reproject (THE BUG-FIX CASES) ────────────────────────────────

    def test_focal_reproject_to_memory_band_count(self, tif_4band):
        """focal().reproject() must return N bands, not 1."""
        res = curaster.open(tif_4band).focal().reproject(TARGET_CRS).to_memory()
        assert res.bands == BANDS, (
            f"focal().reproject().to_memory() returned {res.bands} bands; expected {BANDS}"
        )

    def test_focal_reproject_save_local_band_count(self, tif_4band, tmp_path):
        """focal().reproject().save_local() must write N bands to disk."""
        out = str(tmp_path / "out_focal_reproj.tif")
        curaster.open(tif_4band).focal().reproject(TARGET_CRS).save_local(out)
        n = _band_count_from_file(out)
        assert n == BANDS, (
            f"focal().reproject().save_local() wrote {n} bands; expected {BANDS}"
        )

    def test_focal_reproject_iter_begin_band_count(self, tif_4band):
        arr = _collect_iter(curaster.open(tif_4band).focal().reproject(TARGET_CRS))
        assert arr.shape[0] == BANDS

    def test_focal_clip_reproject_save_local(self, tif_4band, aoi_src, tmp_path):
        """focal().clip().reproject() two-stage path."""
        out = str(tmp_path / "out_focal_cr.tif")
        curaster.open(tif_4band).focal().clip(aoi_src).reproject(TARGET_CRS).save_local(out)
        assert _band_count_from_file(out) == BANDS

    def test_focal_reproject_clip_save_local(self, tif_4band, aoi_dst, tmp_path):
        """focal().reproject().clip() two-stage + target-CRS clip."""
        out = str(tmp_path / "out_focal_rc.tif")
        curaster.open(tif_4band).focal().reproject(TARGET_CRS).clip(aoi_dst).save_local(out)
        assert _band_count_from_file(out) == BANDS

    def test_focal_clip_reproject_ordering_differs(self, tif_4band, aoi_src):
        """
        focal().clip().reproject() and focal().reproject().clip() with the
        same source-CRS polygon should yield different non-zero pixel counts.
        """
        res_cr = (curaster.open(tif_4band).focal()
                  .clip(aoi_src).reproject(TARGET_CRS).to_memory())
        res_rc = (curaster.open(tif_4band).focal()
                  .reproject(TARGET_CRS).clip(aoi_src).to_memory())
        nz_cr = np.count_nonzero(_result_to_array(res_cr)[0])
        nz_rc = np.count_nonzero(_result_to_array(res_rc)[0])
        assert nz_cr != nz_rc


# ── focal + algebra (post-neighborhood algebra) ───────────────────────────────

class TestFocalAlgebra:
    def test_focal_algebra_produces_1_band(self, tif_4band):
        res = curaster.open(tif_4band).focal().algebra("B1 + B2").to_memory()
        assert res.bands == 1

    def test_algebra_focal_produces_1_band(self, tif_4band):
        """Pre-focal algebra → 1 band input → focal → 1 band output."""
        res = curaster.open(tif_4band).algebra("B1 + B2").focal().to_memory()
        assert res.bands == 1

    def test_focal_algebra_save_local(self, tif_4band, tmp_path):
        out = str(tmp_path / "out_focal_alg.tif")
        curaster.open(tif_4band).focal().algebra("B1 * 2").save_local(out)
        assert _band_count_from_file(out) == 1


# ── terrain ───────────────────────────────────────────────────────────────────

class TestTerrain:
    def test_terrain_slope_aspect_2_bands(self, tif_1band):
        res = curaster.open(tif_1band).terrain(metrics=["slope", "aspect"]).to_memory()
        assert res.bands == 2

    def test_terrain_hillshade_1_band(self, tif_1band):
        res = curaster.open(tif_1band).terrain(metrics=["hillshade"]).to_memory()
        assert res.bands == 1

    def test_terrain_save_local(self, tif_1band, tmp_path):
        out = str(tmp_path / "out_terrain.tif")
        curaster.open(tif_1band).terrain(metrics=["slope", "aspect"]).save_local(out)
        assert _band_count_from_file(out) == 2

    def test_terrain_after_algebra(self, tif_4band, tmp_path):
        """algebra → terrain: chain pre-op + neighborhood."""
        out = str(tmp_path / "out_alg_terrain.tif")
        (curaster.open(tif_4band)
            .algebra("(B4 - B3) / (B4 + B3)")
            .terrain(metrics=["slope"])
            .save_local(out))
        assert _band_count_from_file(out) == 1

    def test_terrain_reproject(self, tif_1band, tmp_path):
        """terrain().reproject(): two-stage path for single-band terrain."""
        out = str(tmp_path / "out_terrain_reproj.tif")
        (curaster.open(tif_1band)
            .terrain(metrics=["slope", "aspect"])
            .reproject(TARGET_CRS)
            .save_local(out))
        assert _band_count_from_file(out) == 2


# ── pre-op + neighborhood combos ──────────────────────────────────────────────

class TestPreOpNeighborhood:
    def test_clip_focal_multiband(self, tif_4band, aoi_src):
        """clip() before focal: pre-op clips in source CRS, focal runs on clipped data."""
        res = (curaster.open(tif_4band)
               .clip(aoi_src)
               .focal()
               .to_memory())
        assert res.bands == BANDS

    def test_reproject_focal(self, tif_4band):
        """reproject() before focal: focal runs in target CRS."""
        res = (curaster.open(tif_4band)
               .reproject(TARGET_CRS)
               .focal()
               .to_memory())
        assert res.bands == BANDS

    def test_algebra_focal_clip(self, tif_4band, aoi_src):
        res = (curaster.open(tif_4band)
               .algebra("B1 + B2")
               .focal()
               .clip(aoi_src)
               .to_memory())
        assert res.bands == 1


# ── VRAM cache (persist) ──────────────────────────────────────────────────────

class TestPersist:
    def test_persist_passthrough(self, tif_4band):
        res = curaster.open(tif_4band).persist().to_memory()
        assert res.bands == BANDS

    def test_persist_clip(self, tif_4band, aoi_src):
        res = curaster.open(tif_4band).persist().clip(aoi_src).to_memory()
        assert res.bands == BANDS

    def test_persist_algebra(self, tif_4band):
        res = curaster.open(tif_4band).persist().algebra("B1 + B2").to_memory()
        assert res.bands == 1

    def test_persist_reproject(self, tif_4band):
        res = curaster.open(tif_4band).persist().reproject(TARGET_CRS).to_memory()
        assert res.bands == BANDS


# ── save_s3 (local temp-file path, no real S3 needed) ────────────────────────

class TestSaveS3:
    """
    save_s3 writes to a /tmp file and then CPLCopyFile to the s3_path.
    We pass a local path prefixed with /vsimem/ to avoid needing real AWS creds
    while still exercising the save_local → copy logic.
    NOTE: This will fail if the target path is not writable; skip gracefully.
    """

    def _vsimem_path(self, name: str) -> str:
        return f"/vsimem/curaster_test_{name}.tif"

    def test_focal_clip_save_s3_band_count(self, tif_4band, aoi_src):
        vsi = self._vsimem_path("focal_clip_s3")
        try:
            curaster.open(tif_4band).focal().clip(aoi_src).save_s3(vsi)
        except RuntimeError as e:
            if "S3 upload failed" in str(e):
                pytest.skip("S3 upload not available in this environment")
            raise
        from osgeo import gdal as _gdal
        ds = _gdal.Open(vsi)
        assert ds is not None
        n = ds.RasterCount
        ds = None
        from osgeo import gdal as _gdal2
        _gdal2.GetDriverByName("GTiff").Delete(vsi)
        assert n == BANDS

    def test_passthrough_save_s3_band_count(self, tif_4band):
        vsi = self._vsimem_path("pass_s3")
        try:
            curaster.open(tif_4band).save_s3(vsi)
        except RuntimeError as e:
            if "S3 upload failed" in str(e):
                pytest.skip("S3 upload not available in this environment")
            raise
        from osgeo import gdal as _gdal
        ds = _gdal.Open(vsi)
        assert ds is not None
        n = ds.RasterCount
        ds = None
        _gdal.GetDriverByName("GTiff").Delete(vsi)
        assert n == BANDS


# ── get_info ──────────────────────────────────────────────────────────────────

class TestGetInfo:
    def test_get_info_fields(self, tif_4band):
        info = curaster.open(tif_4band).get_info()
        assert "width" in info and info["width"] == W
        assert "height" in info and info["height"] == H
        assert "geotransform" in info and len(info["geotransform"]) == 6
        assert "crs" in info

    def test_get_info_after_reproject(self, tif_4band):
        info = curaster.open(tif_4band).reproject(TARGET_CRS).get_info()
        assert TARGET_CRS.split(":")[-1] in info["crs"]
