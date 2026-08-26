#if canImport(Testing)
import Testing
import EmberMl

@Suite("EmberMl Swift Export Tests")
struct EmberMlExportTests {
    @Test("Swift module loads")
    func testSwiftModuleLoads() {
        #expect(Bool(true), "EmberMl swift module imported cleanly")
    }
}
#elseif canImport(XCTest)
import XCTest
import EmberMl

final class EmberMlExportTests: XCTestCase {
    func testSwiftModuleLoads() throws {
        XCTAssertTrue(true, "EmberMl swift module imported cleanly")
    }
}
#endif
