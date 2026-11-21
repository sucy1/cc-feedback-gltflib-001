"""
Structural validation for glTF models.

The checks implemented here are purely structural: they only inspect the model
(JSON) side of a glTF asset — accessor/bufferView geometry, sparse accessor
capacity, animation sampler counts, and skin / primitive attribute consistency.
They do NOT require buffer binary data to be loaded, so they work on a GLTF
instance created with ``load_file_resources=False`` (or with no resources at
all).
"""
from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Any, Dict, List, Optional

from .models import GLTFModel, Accessor, BufferView

# Byte size of each accessor componentType (glTF 2.0 spec, section 3.6.2.2)
COMPONENT_TYPE_BYTE_SIZE: Dict[int, int] = {
    5120: 1,  # BYTE
    5121: 1,  # UNSIGNED_BYTE
    5122: 2,  # SHORT
    5123: 2,  # UNSIGNED_SHORT
    5125: 4,  # UNSIGNED_INT
    5126: 4,  # FLOAT
}

# Number of components per element for each accessor type
ACCESSOR_TYPE_COMPONENT_COUNT: Dict[str, int] = {
    'SCALAR': 1,
    'VEC2': 2,
    'VEC3': 3,
    'VEC4': 4,
    'MAT2': 4,
    'MAT3': 9,
    'MAT4': 16,
}

# Component types allowed for sparse indices
SPARSE_INDICES_COMPONENT_TYPES = {5121, 5123, 5125}  # UNSIGNED_BYTE / UNSIGNED_SHORT / UNSIGNED_INT

CUBICSPLINE = 'CUBICSPLINE'


@dataclass
class ValidationIssue:
    """
    A single structural problem found in a glTF model.

    code: Machine-readable identifier of the check that failed (e.g. ACCESSOR_BOUNDS).
    path: Location of the offending element, e.g. "accessors[2]" or
        "meshes[0].primitives[0].attributes.JOINTS_0".
    message: Human-readable description of the problem.
    """
    code: str
    path: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {'code': self.code, 'path': self.path, 'message': self.message}


@dataclass
class ValidationReport:
    """
    Result of validating a glTF model. ``ok`` is True when no issues were found.
    ``to_dict()`` returns a JSON-serializable representation.
    """
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.issues) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {'ok': self.ok, 'issues': [issue.to_dict() for issue in self.issues]}


class GLTFValidationError(ValueError):
    """
    Raised by GLTF.validate(strict=True) when the model has validation issues.
    The full report is available on the ``report`` attribute.
    """

    def __init__(self, report: ValidationReport):
        self.report = report
        summary = '; '.join(f'{issue.code} at {issue.path}: {issue.message}' for issue in report.issues[:5])
        if len(report.issues) > 5:
            summary += f'; ... and {len(report.issues) - 5} more issue(s)'
        super().__init__(f'glTF validation failed with {len(report.issues)} issue(s): {summary}')


def validate_model(model: GLTFModel) -> ValidationReport:
    """
    Validates the structure of a GLTFModel and returns a ValidationReport.
    """
    validator = _ModelValidator(model)
    return validator.validate()


class _ModelValidator:
    def __init__(self, model: GLTFModel):
        self.model = model
        self.issues: List[ValidationIssue] = []

    def validate(self) -> ValidationReport:
        if self.model is None:
            self._add('MODEL_MISSING', '', 'GLTF instance has no model to validate')
            return ValidationReport(self.issues)
        self._validate_accessors()
        self._validate_animations()
        self._validate_meshes()
        self._validate_skins()
        return ValidationReport(self.issues)

    def _add(self, code: str, path: str, message: str) -> None:
        self.issues.append(ValidationIssue(code=code, path=path, message=message))

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def _validate_accessors(self) -> None:
        for i, accessor in enumerate(self.model.accessors or []):
            path = f'accessors[{i}]'
            element_size = self._accessor_element_size(accessor, path)
            if element_size is None:
                continue  # Unknown componentType / type already reported; bounds cannot be computed.
            if accessor.bufferView is not None:
                self._validate_accessor_bounds(i, accessor, element_size, path)
            if accessor.sparse is not None:
                self._validate_sparse(i, accessor, path)

    def _accessor_element_size(self, accessor: Accessor, path: str) -> Optional[int]:
        """Size in bytes of one accessor element (component size * number of components)."""
        component_size = COMPONENT_TYPE_BYTE_SIZE.get(accessor.componentType)
        if component_size is None:
            self._add('ACCESSOR_COMPONENT_TYPE', path,
                      f'Unknown accessor componentType: {accessor.componentType}')
            return None
        component_count = ACCESSOR_TYPE_COMPONENT_COUNT.get(accessor.type)
        if component_count is None:
            self._add('ACCESSOR_TYPE', path, f'Unknown accessor type: {accessor.type}')
            return None
        return component_size * component_count

    def _get_buffer_view(self, index: int) -> Optional[BufferView]:
        buffer_views = self.model.bufferViews or []
        if index is None or index < 0 or index >= len(buffer_views):
            return None
        return buffer_views[index]

    def _validate_accessor_bounds(self, index: int, accessor: Accessor, element_size: int, path: str) -> None:
        buffer_view = self._get_buffer_view(accessor.bufferView)
        if buffer_view is None:
            self._add('ACCESSOR_BUFFERVIEW_INDEX', path,
                      f'Accessor references bufferView index {accessor.bufferView}, but the model has '
                      f'{len(self.model.bufferViews or [])} bufferView(s)')
            return
        component_size = COMPONENT_TYPE_BYTE_SIZE[accessor.componentType]
        byte_offset = accessor.byteOffset or 0
        if byte_offset % component_size != 0:
            self._add('ACCESSOR_OFFSET_ALIGNMENT', path,
                      f'Accessor byteOffset ({byte_offset}) is not aligned to the size of its componentType '
                      f'({component_size} bytes)')
        count = accessor.count or 0
        if count == 0:
            return
        # Stride comes from the bufferView when interleaved; otherwise elements are tightly packed.
        stride = buffer_view.byteStride if buffer_view.byteStride is not None else element_size
        required = byte_offset + (count - 1) * stride + element_size
        if required > (buffer_view.byteLength or 0):
            self._add('ACCESSOR_BUFFER_BOUNDS', path,
                      f'Accessor requires {required} byte(s) in bufferView {accessor.bufferView} '
                      f'(byteOffset={byte_offset}, count={count}, stride={stride}, elementSize={element_size}), '
                      f'but the bufferView only has byteLength={buffer_view.byteLength}')

    def _validate_sparse(self, index: int, accessor: Accessor, path: str) -> None:
        sparse = accessor.sparse
        count = sparse.count or 0

        indices_path = f'{path}.sparse.indices'
        if sparse.indices is None:
            self._add('SPARSE_INDICES_BOUNDS', indices_path, 'Sparse accessor is missing its indices')
        else:
            indices = sparse.indices
            component_size = COMPONENT_TYPE_BYTE_SIZE.get(indices.componentType)
            if component_size is None or indices.componentType not in SPARSE_INDICES_COMPONENT_TYPES:
                self._add('SPARSE_INDICES_BOUNDS', indices_path,
                          f'Sparse indices have invalid componentType: {indices.componentType} '
                          f'(must be one of {sorted(SPARSE_INDICES_COMPONENT_TYPES)})')
            else:
                buffer_view = self._get_buffer_view(indices.bufferView)
                byte_offset = indices.byteOffset or 0
                if buffer_view is None:
                    self._add('SPARSE_INDICES_BOUNDS', indices_path,
                              f'Sparse indices reference bufferView index {indices.bufferView}, but the model has '
                              f'{len(self.model.bufferViews or [])} bufferView(s)')
                else:
                    required = byte_offset + count * component_size
                    if required > (buffer_view.byteLength or 0):
                        self._add('SPARSE_INDICES_BOUNDS', indices_path,
                                  f'Sparse indices require {required} byte(s) in bufferView {indices.bufferView} '
                                  f'(byteOffset={byte_offset}, count={count}, componentSize={component_size}), '
                                  f'but the bufferView only has byteLength={buffer_view.byteLength}')

        values_path = f'{path}.sparse.values'
        if sparse.values is None:
            self._add('SPARSE_VALUES_BOUNDS', values_path, 'Sparse accessor is missing its values')
        else:
            values = sparse.values
            buffer_view = self._get_buffer_view(values.bufferView)
            byte_offset = values.byteOffset or 0
            if buffer_view is None:
                self._add('SPARSE_VALUES_BOUNDS', values_path,
                          f'Sparse values reference bufferView index {values.bufferView}, but the model has '
                          f'{len(self.model.bufferViews or [])} bufferView(s)')
            else:
                # Conservatively reserve 4 bytes per component (the maximum component size) for sparse
                # values, so the capacity check does not under-reserve if the accessor's componentType
                # is widened later.
                component_count = ACCESSOR_TYPE_COMPONENT_COUNT[accessor.type]
                values_element_size = component_count * 4
                required = byte_offset + count * values_element_size
                if required > (buffer_view.byteLength or 0):
                    self._add('SPARSE_VALUES_BOUNDS', values_path,
                              f'Sparse values require {required} byte(s) in bufferView {values.bufferView} '
                              f'(byteOffset={byte_offset}, count={count}, elementSize={values_element_size}), '
                              f'but the bufferView only has byteLength={buffer_view.byteLength}')

    # ------------------------------------------------------------------
    # Animations
    # ------------------------------------------------------------------

    def _validate_animations(self) -> None:
        accessors = self.model.accessors or []
        for i, animation in enumerate(self.model.animations or []):
            for j, sampler in enumerate(animation.samplers or []):
                path = f'animations[{i}].samplers[{j}]'
                input_accessor = self._get_accessor(sampler.input)
                if input_accessor is None:
                    self._add('ANIMATION_INPUT_INDEX', f'{path}.input',
                              f'Sampler input references accessor index {sampler.input}, but the model has '
                              f'{len(accessors)} accessor(s)')
                    continue
                output_accessor = self._get_accessor(sampler.output)
                if output_accessor is None:
                    self._add('ANIMATION_OUTPUT_INDEX', f'{path}.output',
                              f'Sampler output references accessor index {sampler.output}, but the model has '
                              f'{len(accessors)} accessor(s)')
                    continue
                input_count = input_accessor.count or 0
                output_count = output_accessor.count or 0
                if sampler.interpolation == CUBICSPLINE:
                    if output_count != 3 * input_count:
                        self._add('ANIMATION_CUBICSPLINE_COUNT', f'{path}.output',
                                  f'CUBICSPLINE interpolation requires output.count ({output_count}) to be 3 times '
                                  f'input.count ({input_count})')
                elif output_count != input_count:
                    self._add('ANIMATION_OUTPUT_COUNT', f'{path}.output',
                              f'Sampler output.count ({output_count}) does not match input.count ({input_count})')

    # ------------------------------------------------------------------
    # Meshes (JOINTS_n / WEIGHTS_n pairing)
    # ------------------------------------------------------------------

    def _validate_meshes(self) -> None:
        for i, mesh in enumerate(self.model.meshes or []):
            for j, primitive in enumerate(mesh.primitives or []):
                attributes = primitive.attributes
                if attributes is None:
                    continue
                values = self._attribute_values(attributes)
                path = f'meshes[{i}].primitives[{j}].attributes'
                for name, accessor_index in values.items():
                    if self._get_accessor(accessor_index) is None:
                        self._add('ATTRIBUTE_ACCESSOR_INDEX', f'{path}.{name}',
                                  f'Attribute {name} references accessor index {accessor_index}, but the model has '
                                  f'{len(self.model.accessors or [])} accessor(s)')
                suffixes = set()
                for name in values:
                    for prefix in ('JOINTS_', 'WEIGHTS_'):
                        if name.startswith(prefix):
                            suffixes.add(name[len(prefix):])
                for suffix in sorted(suffixes):
                    joints_name, weights_name = f'JOINTS_{suffix}', f'WEIGHTS_{suffix}'
                    if joints_name not in values or weights_name not in values:
                        continue
                    joints_accessor = self._get_accessor(values[joints_name])
                    weights_accessor = self._get_accessor(values[weights_name])
                    if joints_accessor is None or weights_accessor is None:
                        continue  # Out-of-range accessor indices already reported above.
                    if (joints_accessor.count or 0) != (weights_accessor.count or 0):
                        self._add('ATTRIBUTE_COUNT_MISMATCH', f'{path}.{joints_name}',
                                  f'{joints_name}.count ({joints_accessor.count}) does not match '
                                  f'{weights_name}.count ({weights_accessor.count})')

    @staticmethod
    def _attribute_values(attributes: Any) -> Dict[str, int]:
        if isinstance(attributes, dict):
            return {k: v for k, v in attributes.items() if isinstance(v, int)}
        return {f.name: getattr(attributes, f.name)
                for f in dataclass_fields(attributes)
                if isinstance(getattr(attributes, f.name), int)}

    # ------------------------------------------------------------------
    # Skins
    # ------------------------------------------------------------------

    def _validate_skins(self) -> None:
        for i, skin in enumerate(self.model.skins or []):
            path = f'skins[{i}]'
            if skin.inverseBindMatrices is None:
                continue
            ibm_accessor = self._get_accessor(skin.inverseBindMatrices)
            if ibm_accessor is None:
                self._add('SKIN_IBM_INDEX', f'{path}.inverseBindMatrices',
                          f'Skin inverseBindMatrices references accessor index {skin.inverseBindMatrices}, but the '
                          f'model has {len(self.model.accessors or [])} accessor(s)')
                continue
            joint_count = len(skin.joints or [])
            ibm_count = ibm_accessor.count or 0
            if ibm_accessor.type != 'MAT4':
                self._add('SKIN_IBM_COUNT', path,
                          f'Skin inverseBindMatrices accessor has type {ibm_accessor.type}, but must be MAT4 '
                          f'to hold one 4x4 matrix per joint ({joint_count} joint(s))')
            elif joint_count > ibm_count:
                self._add('SKIN_IBM_COUNT', path,
                          f'Skin has {joint_count} joint(s) but inverseBindMatrices accessor only has '
                          f'count={ibm_count}')

    # ------------------------------------------------------------------

    def _get_accessor(self, index: Optional[int]) -> Optional[Accessor]:
        accessors = self.model.accessors or []
        if index is None or index < 0 or index >= len(accessors):
            return None
        return accessors[index]
