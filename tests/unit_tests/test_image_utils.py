import base64
import json

import pytest

from energyevals.agent.schema import ImageContent
from energyevals.utils.image_utils import (
    decode_base64_to_bytes,
    encode_image_to_base64,
    extract_images_from_result,
)


class TestExtractImagesFromResult:
    """Tests for extract_images_from_result function."""

    def test_extract_single_base64_image(self):
        """Test extracting a single base64 image."""
        result = json.dumps({
            "image_base64": "iVBORw0KGgoAAAANSUhEUg==",
            "media_type": "image/png",
        })

        images = extract_images_from_result(result)

        assert len(images) == 1
        assert isinstance(images[0], ImageContent)
        assert images[0].image_base64 == "iVBORw0KGgoAAAANSUhEUg=="
        assert images[0].media_type == "image/png"

    def test_extract_image_data_format(self):
        """Test extracting image with image_data key."""
        result = json.dumps({
            "image_data": "base64encodeddata",
            "media_type": "image/jpeg",
        })

        images = extract_images_from_result(result)

        assert len(images) == 1
        assert images[0].image_base64 == "base64encodeddata"

    def test_extract_images_from_list(self):
        """Test extracting images from list format."""
        result = json.dumps({
            "images": [
                {"image_base64": "img1data", "media_type": "image/png"},
                {"image_base64": "img2data", "media_type": "image/jpeg"},
            ]
        })

        images = extract_images_from_result(result)

        assert len(images) == 2
        assert images[0].image_base64 == "img1data"
        assert images[1].image_base64 == "img2data"

    def test_extract_images_from_list_data_key(self):
        """Test extracting images from list with 'data' key."""
        result = json.dumps({
            "images": [
                {"data": "img1data"},
                {"data": "img2data"},
            ]
        })

        images = extract_images_from_result(result)

        assert len(images) == 2

    def test_no_images_found(self):
        """Test when no images are present."""
        result = json.dumps({"data": "text only", "value": 123})

        images = extract_images_from_result(result)

        assert len(images) == 0

    def test_non_json_string(self):
        """Test with plain text (non-JSON)."""
        result = "Just plain text with no images"

        images = extract_images_from_result(result)

        assert len(images) == 0

    def test_default_media_type(self):
        """Test that default media type is image/jpeg."""
        result = json.dumps({"image_base64": "somedata"})

        images = extract_images_from_result(result)

        assert len(images) == 1
        assert images[0].media_type == "image/jpeg"


class TestEncodeImageToBase64:
    """Tests for encode_image_to_base64 function."""

    def test_encode_simple_file(self, tmp_path):
        """Test encoding a simple file to base64."""
        test_file = tmp_path / "test.txt"
        test_content = b"test image data"
        test_file.write_bytes(test_content)

        encoded = encode_image_to_base64(str(test_file))

        assert isinstance(encoded, str)
        decoded = base64.b64decode(encoded)
        assert decoded == test_content

    def test_encode_binary_data(self, tmp_path):
        """Test encoding binary image data."""
        test_file = tmp_path / "image.bin"
        binary_data = bytes([0xFF, 0xD8, 0xFF, 0xE0])  # Fake JPEG header
        test_file.write_bytes(binary_data)

        encoded = encode_image_to_base64(str(test_file))
        decoded = base64.b64decode(encoded)

        assert decoded == binary_data

    def test_encode_nonexistent_file(self):
        """Test encoding a file that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            encode_image_to_base64("/nonexistent/file.jpg")

    def test_encode_empty_file(self, tmp_path):
        """Test encoding an empty file."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_bytes(b"")

        encoded = encode_image_to_base64(str(empty_file))

        assert encoded == ""


class TestDecodeBase64ToBytes:
    """Tests for decode_base64_to_bytes function."""

    def test_decode_simple_base64(self):
        """Test decoding simple base64 string."""
        original = b"test data"
        encoded = base64.b64encode(original).decode("utf-8")

        decoded = decode_base64_to_bytes(encoded)

        assert decoded == original

    def test_decode_binary_data(self):
        """Test decoding base64 with binary data."""
        original = bytes([0x89, 0x50, 0x4E, 0x47])  # PNG header
        encoded = base64.b64encode(original).decode("utf-8")

        decoded = decode_base64_to_bytes(encoded)

        assert decoded == original

    def test_roundtrip_encoding(self, tmp_path):
        """Test encoding and decoding roundtrip."""
        test_file = tmp_path / "test.bin"
        original_data = b"round trip test data"
        test_file.write_bytes(original_data)

        encoded = encode_image_to_base64(str(test_file))
        decoded = decode_base64_to_bytes(encoded)

        assert decoded == original_data

    def test_decode_invalid_base64(self):
        """Test decoding invalid base64 string."""
        with pytest.raises(Exception):  # base64 decode error
            decode_base64_to_bytes("not valid base64!!!")

    def test_decode_empty_string(self):
        """Test decoding empty string."""
        decoded = decode_base64_to_bytes("")
        assert decoded == b""


class TestImageContentCreation:
    """Tests for ImageContent object creation from extracted data."""

    def test_image_content_attributes(self):
        """Test that extracted images have correct attributes."""
        result = json.dumps({"image_base64": "testdata"})

        images = extract_images_from_result(result)

        assert len(images) == 1
        img = images[0]
        assert hasattr(img, "image_base64")
        assert hasattr(img, "type")
        assert img.type == "image"
