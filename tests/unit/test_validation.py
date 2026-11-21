import json
from unittest import TestCase

from gltflib import (
    GLTF, GLTFModel, Asset, Accessor, Buffer, BufferView, Sparse, SparseIndices, SparseValues,
    Animation, AnimationSampler, Mesh, Primitive, Attributes, Skin,
    ValidationReport, ValidationIssue, GLTFValidationError)

FLOAT = 5126
UNSIGNED_SHORT = 5123
UNSIGNED_INT = 5125


def make_gltf(**kwargs):
    return GLTF(model=GLTFModel(asset=Asset(version='2.0'), **kwargs))


def codes(report):
    return [issue.code for issue in report.issues]


class TestAccessorValidation(TestCase):
    def test_valid_model_has_no_issues(self):
        gltf = make_gltf(
            buffers=[Buffer(byteLength=24)],
            bufferViews=[BufferView(buffer=0, byteLength=24)],
            accessors=[Accessor(bufferView=0, componentType=FLOAT, count=2, type='VEC3')])
        report = gltf.validate()
        self.assertTrue(report.ok)
        self.assertEqual([], report.issues)

    def test_accessor_bounds_tightly_packed(self):
        # 2 VEC3 floats = 24 bytes required, but bufferView only has 20
        gltf = make_gltf(
            buffers=[Buffer(byteLength=20)],
            bufferViews=[BufferView(buffer=0, byteLength=20)],
            accessors=[Accessor(bufferView=0, componentType=FLOAT, count=2, type='VEC3')])
        report = gltf.validate()
        self.assertFalse(report.ok)
        self.assertEqual(['ACCESSOR_BUFFER_BOUNDS'], codes(report))
        self.assertEqual('accessors[0]', report.issues[0].path)

    def test_accessor_bounds_with_byte_offset(self):
        # byteOffset=8, 1 VEC3 float = 12 bytes -> 20 required, only 16 available
        gltf = make_gltf(
            buffers=[Buffer(byteLength=16)],
            bufferViews=[BufferView(buffer=0, byteLength=16)],
            accessors=[Accessor(bufferView=0, byteOffset=8, componentType=FLOAT, count=1, type='VEC3')])
        report = gltf.validate()
        self.assertIn('ACCESSOR_BUFFER_BOUNDS', codes(report))

    def test_accessor_bounds_respects_buffer_view_byte_stride(self):
        # Interleaved: stride 16, count 2 -> (2-1)*16 + 12 = 28 bytes required.
        # Tightly-packed assumption (24) would wrongly pass a 26-byte view.
        gltf = make_gltf(
            buffers=[Buffer(byteLength=26)],
            bufferViews=[BufferView(buffer=0, byteLength=26, byteStride=16)],
            accessors=[Accessor(bufferView=0, componentType=FLOAT, count=2, type='VEC3')])
        report = gltf.validate()
        self.assertEqual(['ACCESSOR_BUFFER_BOUNDS'], codes(report))

        gltf.model.bufferViews[0].byteLength = 28
        self.assertTrue(gltf.validate().ok)

    def test_accessor_byteoffset_alignment(self):
        # FLOAT components require 4-byte alignment
        gltf = make_gltf(
            buffers=[Buffer(byteLength=32)],
            bufferViews=[BufferView(buffer=0, byteLength=32)],
            accessors=[Accessor(bufferView=0, byteOffset=2, componentType=FLOAT, count=1, type='SCALAR')])
        report = gltf.validate()
        self.assertIn('ACCESSOR_OFFSET_ALIGNMENT', codes(report))
        self.assertEqual('accessors[0]', report.issues[0].path)

    def test_accessor_byteoffset_alignment_ok_for_byte_components(self):
        gltf = make_gltf(
            buffers=[Buffer(byteLength=8)],
            bufferViews=[BufferView(buffer=0, byteLength=8)],
            accessors=[Accessor(bufferView=0, byteOffset=3, componentType=5121, count=1, type='SCALAR')])
        self.assertTrue(gltf.validate().ok)

    def test_accessor_buffer_view_index_out_of_range(self):
        gltf = make_gltf(
            buffers=[Buffer(byteLength=8)],
            bufferViews=[BufferView(buffer=0, byteLength=8)],
            accessors=[Accessor(bufferView=5, componentType=FLOAT, count=1, type='SCALAR')])
        report = gltf.validate()
        self.assertEqual(['ACCESSOR_BUFFERVIEW_INDEX'], codes(report))

    def test_accessor_without_buffer_view_is_allowed(self):
        gltf = make_gltf(accessors=[Accessor(componentType=FLOAT, count=3, type='VEC3')])
        self.assertTrue(gltf.validate().ok)


class TestSparseValidation(TestCase):
    def make_sparse_gltf(self, indices_byte_length=6, values_byte_length=12, sparse_count=3):
        return make_gltf(
            buffers=[Buffer(byteLength=64)],
            bufferViews=[
                BufferView(buffer=0, byteLength=40),
                BufferView(buffer=0, byteLength=indices_byte_length),
                BufferView(buffer=0, byteLength=values_byte_length)],
            accessors=[Accessor(
                bufferView=0, componentType=FLOAT, count=10, type='SCALAR',
                sparse=Sparse(
                    count=sparse_count,
                    indices=SparseIndices(bufferView=1, byteOffset=0, componentType=UNSIGNED_SHORT),
                    values=SparseValues(bufferView=2, byteOffset=0)))])

    def test_valid_sparse_accessor(self):
        self.assertTrue(self.make_sparse_gltf().validate().ok)

    def test_sparse_indices_bounds(self):
        # 3 UNSIGNED_SHORT indices need 6 bytes; only 4 available
        report = self.make_sparse_gltf(indices_byte_length=4).validate()
        self.assertEqual(['SPARSE_INDICES_BOUNDS'], codes(report))
        self.assertEqual('accessors[0].sparse.indices', report.issues[0].path)

    def test_sparse_values_bounds(self):
        # 3 SCALAR float values need 12 bytes; only 8 available
        report = self.make_sparse_gltf(values_byte_length=8).validate()
        self.assertEqual(['SPARSE_VALUES_BOUNDS'], codes(report))
        self.assertEqual('accessors[0].sparse.values', report.issues[0].path)

    def test_sparse_indices_and_values_both_out_of_bounds(self):
        report = self.make_sparse_gltf(indices_byte_length=2, values_byte_length=4).validate()
        self.assertEqual(['SPARSE_INDICES_BOUNDS', 'SPARSE_VALUES_BOUNDS'], codes(report))

    def test_sparse_indices_invalid_component_type(self):
        gltf = self.make_sparse_gltf()
        gltf.model.accessors[0].sparse.indices.componentType = FLOAT
        report = gltf.validate()
        self.assertIn('SPARSE_INDICES_BOUNDS', codes(report))

    def test_sparse_buffer_view_index_out_of_range(self):
        gltf = self.make_sparse_gltf()
        gltf.model.accessors[0].sparse.indices.bufferView = 9
        gltf.model.accessors[0].sparse.values.bufferView = 9
        report = gltf.validate()
        self.assertEqual(['SPARSE_INDICES_BOUNDS', 'SPARSE_VALUES_BOUNDS'], codes(report))


class TestAnimationValidation(TestCase):
    def make_animation_gltf(self, input_count=4, output_count=4, interpolation='LINEAR'):
        return make_gltf(
            accessors=[
                Accessor(componentType=FLOAT, count=input_count, type='SCALAR'),
                Accessor(componentType=FLOAT, count=output_count, type='VEC3')],
            animations=[Animation(
                channels=[],
                samplers=[AnimationSampler(input=0, output=1, interpolation=interpolation)])])

    def test_valid_animation(self):
        self.assertTrue(self.make_animation_gltf().validate().ok)

    def test_animation_output_count_mismatch(self):
        report = self.make_animation_gltf(input_count=4, output_count=3).validate()
        self.assertEqual(['ANIMATION_OUTPUT_COUNT'], codes(report))
        self.assertEqual('animations[0].samplers[0].output', report.issues[0].path)

    def test_animation_cubicspline_count(self):
        # CUBICSPLINE requires output.count == 3 * input.count
        report = self.make_animation_gltf(input_count=4, output_count=4, interpolation='CUBICSPLINE').validate()
        self.assertEqual(['ANIMATION_CUBICSPLINE_COUNT'], codes(report))

    def test_animation_cubicspline_valid(self):
        self.assertTrue(self.make_animation_gltf(input_count=4, output_count=12,
                                                 interpolation='CUBICSPLINE').validate().ok)

    def test_animation_accessor_index_out_of_range(self):
        gltf = self.make_animation_gltf()
        gltf.model.animations[0].samplers[0].input = 7
        gltf.model.animations[0].samplers[0].output = 8
        report = gltf.validate()
        self.assertEqual(['ANIMATION_INPUT_INDEX'], codes(report))


class TestSkinAndAttributeValidation(TestCase):
    def make_skinned_gltf(self, joints_count=4, weights_count=4, num_joints=2, ibm_count=2):
        return make_gltf(
            accessors=[
                Accessor(componentType=UNSIGNED_SHORT, count=joints_count, type='VEC4'),
                Accessor(componentType=FLOAT, count=weights_count, type='VEC4'),
                Accessor(componentType=FLOAT, count=ibm_count, type='MAT4')],
            meshes=[Mesh(primitives=[Primitive(
                attributes=Attributes(POSITION=None, JOINTS_0=0, WEIGHTS_0=1))])],
            skins=[Skin(inverseBindMatrices=2, joints=list(range(num_joints)))])

    def test_valid_skinned_model(self):
        self.assertTrue(self.make_skinned_gltf().validate().ok)

    def test_joints_weights_count_mismatch(self):
        report = self.make_skinned_gltf(joints_count=4, weights_count=3).validate()
        self.assertEqual(['ATTRIBUTE_COUNT_MISMATCH'], codes(report))
        self.assertEqual('meshes[0].primitives[0].attributes.JOINTS_0', report.issues[0].path)

    def test_skin_ibm_count(self):
        report = self.make_skinned_gltf(num_joints=3, ibm_count=2).validate()
        self.assertEqual(['SKIN_IBM_COUNT'], codes(report))
        self.assertEqual('skins[0]', report.issues[0].path)

    def test_skin_ibm_count_equal_is_ok(self):
        self.assertTrue(self.make_skinned_gltf(num_joints=2, ibm_count=2).validate().ok)

    def test_skin_ibm_accessor_must_be_mat4(self):
        gltf = self.make_skinned_gltf(num_joints=2, ibm_count=2)
        gltf.model.accessors[2].type = 'VEC3'
        report = gltf.validate()
        self.assertEqual(['SKIN_IBM_COUNT'], codes(report))
        self.assertEqual('skins[0]', report.issues[0].path)

    def test_skin_without_inverse_bind_matrices_is_ok(self):
        gltf = self.make_skinned_gltf()
        gltf.model.skins[0].inverseBindMatrices = None
        self.assertTrue(gltf.validate().ok)


class TestValidationReportAndStrictMode(TestCase):
    def make_invalid_gltf(self):
        return make_gltf(
            buffers=[Buffer(byteLength=4)],
            bufferViews=[BufferView(buffer=0, byteLength=4)],
            accessors=[Accessor(bufferView=0, componentType=FLOAT, count=2, type='VEC3')])

    def test_report_is_json_serializable(self):
        report = self.make_invalid_gltf().validate()
        data = json.loads(json.dumps(report.to_dict()))
        self.assertFalse(data['ok'])
        self.assertEqual(1, len(data['issues']))
        issue = data['issues'][0]
        self.assertEqual('ACCESSOR_BUFFER_BOUNDS', issue['code'])
        self.assertEqual('accessors[0]', issue['path'])
        self.assertIn('byteLength', issue['message'])

    def test_empty_report_is_json_serializable(self):
        report = make_gltf().validate()
        self.assertEqual({'ok': True, 'issues': []}, json.loads(json.dumps(report.to_dict())))

    def test_strict_raises_with_report(self):
        gltf = self.make_invalid_gltf()
        with self.assertRaises(GLTFValidationError) as ctx:
            gltf.validate(strict=True)
        self.assertIsInstance(ctx.exception, ValueError)
        self.assertIsInstance(ctx.exception.report, ValidationReport)
        self.assertEqual(['ACCESSOR_BUFFER_BOUNDS'], codes(ctx.exception.report))

    def test_non_strict_does_not_raise(self):
        report = self.make_invalid_gltf().validate(strict=False)
        self.assertFalse(report.ok)

    def test_strict_valid_model_returns_report(self):
        report = make_gltf().validate(strict=True)
        self.assertTrue(report.ok)

    def test_validation_issue_fields(self):
        issue = ValidationIssue(code='X', path='accessors[0]', message='msg')
        self.assertEqual({'code': 'X', 'path': 'accessors[0]', 'message': 'msg'}, issue.to_dict())


class TestValidationWithoutResources(TestCase):
    def test_validate_loaded_gltf_without_file_resources(self):
        """Capacity checks must run from bufferView metadata alone, without any buffer binary loaded."""
        model_json = json.dumps({
            'asset': {'version': '2.0'},
            'buffers': [{'uri': 'buffer.bin', 'byteLength': 8}],
            'bufferViews': [{'buffer': 0, 'byteOffset': 0, 'byteLength': 8}],
            'accessors': [{'bufferView': 0, 'componentType': FLOAT, 'count': 2, 'type': 'VEC3'}]
        })
        import io
        gltf = GLTF.read_gltf(io.BytesIO(model_json.encode('utf-8')), load_file_resources=False)
        # Buffer binary was never loaded, but the bounds check must still fire.
        report = gltf.validate()
        self.assertFalse(report.ok)
        self.assertEqual(['ACCESSOR_BUFFER_BOUNDS'], codes(report))
