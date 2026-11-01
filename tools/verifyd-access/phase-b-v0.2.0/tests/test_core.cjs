const assert = require('node:assert/strict');
const core = require('../extension/navigation-core.js');
assert.equal(core.nextIndex(-1, 3), 0);
assert.equal(core.nextIndex(2, 3), 0);
assert.equal(core.nextIndex(-1, 0), -1);
assert.deepEqual(core.movePoint({x:10, y:10},'left',50,{width:200,height:100}), {x:4,y:10});
assert.deepEqual(core.movePoint({x:10, y:10},'down',500,{width:200,height:100}), {x:10,y:96});
assert.equal(core.cleanLabel('   too  many\nspaces '),'too many spaces');
console.log('PASS navigation-core unit tests');
